use crate::lazybuffer::{Backend, LazyBuffer};
use std::collections::VecDeque;
use std::panic;
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub buffer: LazyBuffer,
    pub gradient: Option<LazyBuffer>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        Tensor {
            buffer: LazyBuffer::new(data),
            gradient: None,
            requires_grad: true,
        }
    }
    pub fn without_grad(data: Vec<f32>) -> Self {
        Tensor {
            buffer: LazyBuffer::new(data),
            gradient: None,
            requires_grad: false,
        }
    }
    pub fn realize(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, false);
    }
    pub fn realize_to_host(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, true);
    }
    /// Begin a training loop with buffer caching
    pub fn begin_training_loop(backend: &dyn Backend) {
        crate::training::begin_training_loop(backend);
    }

    /// End a training loop and free cached buffers
    pub fn end_training_loop(backend_name: &str) {
        crate::training::end_training_loop(backend_name);
    }
    pub fn apply_gradients(&mut self, backend: &dyn Backend) {
        if let Some(grad) = &self.gradient {
            self.buffer = LazyBuffer::from_operation(TensorOperation::Subtract(
                Arc::new(self.clone()),
                Arc::new(Tensor {
                    buffer: grad.clone(),
                    gradient: None,
                    requires_grad: false,
                }),
            ));
            self.buffer.realize(backend, true);
            // self.gradient = None;
        } else {
            
            panic!("No gradient available to apply.");

        }
    }
    pub fn backward(&mut self, backend: &dyn Backend) {
        let mut queue = VecDeque::<(Tensor, LazyBuffer)>::new();
        queue.push_back((self.clone(), LazyBuffer::new(vec![1.0; self.buffer.size])));
        self.gradient = Some(LazyBuffer::new(vec![1.0; self.buffer.size]));
        let mut all_tensors_involved = Vec::new();
        while let Some((curr_tensor, curr_gradient)) = queue.pop_front() {
            if curr_tensor.requires_grad {
                all_tensors_involved.push(curr_tensor.clone());
                match &curr_tensor.buffer.operation {
                    TensorOperation::Add(left, right) => {
                        // Both operands receive the same gradient in addition
                        let mut left_tensor = (**left).clone();
                        let mut right_tensor = (**right).clone();

                        left_tensor.gradient = Some(curr_gradient.clone());
                        right_tensor.gradient = Some(curr_gradient.clone());

                        queue.push_back((left_tensor, curr_gradient.clone()));
                        queue.push_back((right_tensor, curr_gradient));
                    }
                    TensorOperation::Multiply(left, right) => {
                        // Chain rule for multiplication: left gets right * gradient, right gets left * gradient
                        let mut left_tensor = (**left).clone();
                        let mut right_tensor = (**right).clone();

                        let left_grad = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: right_tensor.buffer.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(Tensor {
                                buffer: curr_gradient.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                        ));
                        let right_grad = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: left_tensor.buffer.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(Tensor {
                                buffer: curr_gradient.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                        ));

                        left_tensor.gradient = Some(left_grad.clone());
                        right_tensor.gradient = Some(right_grad.clone());

                        queue.push_back((left_tensor, left_grad));
                        queue.push_back((right_tensor, right_grad));
                    }
                    TensorOperation::Divide(left, right) => {
                        // Chain rule for division:
                        // left gets gradient/right
                        // right gets -left*gradient/(right*right)
                        let mut left_tensor = (**left).clone();
                        let mut right_tensor = (**right).clone();

                        let left_grad = LazyBuffer::from_operation(TensorOperation::Divide(
                            Arc::new(Tensor {
                                buffer: curr_gradient.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            right.clone(),
                        ));

                        // For right gradient: first compute -gradient
                        let neg_one = Tensor {
                            buffer: LazyBuffer::new(vec![-1.0; curr_gradient.size]),
                            gradient: None,
                            requires_grad: false,
                        };
                        let neg_grad = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: curr_gradient.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(neg_one),
                        ));

                        // Then multiply by left
                        let temp = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: left_tensor.buffer.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(Tensor {
                                buffer: neg_grad,
                                gradient: None,
                                requires_grad: false,
                            }),
                        ));

                        // Finally divide by right squared
                        let right_squared = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: right_tensor.buffer.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(Tensor {
                                buffer: right_tensor.buffer.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                        ));
                        let right_grad = LazyBuffer::from_operation(TensorOperation::Divide(
                            Arc::new(Tensor {
                                buffer: temp,
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(Tensor {
                                buffer: right_squared,
                                gradient: None,
                                requires_grad: false,
                            }),
                        ));

                        left_tensor.gradient = Some(left_grad.clone());
                        right_tensor.gradient = Some(right_grad.clone());

                        queue.push_back((left_tensor, left_grad));
                        queue.push_back((right_tensor, right_grad));
                    }
                    TensorOperation::Subtract(left, right) => {
                        // Left gets gradient, right gets -gradient
                        let mut left_tensor = (**left).clone();
                        let mut right_tensor = (**right).clone();

                        let neg_one = Tensor {
                            buffer: LazyBuffer::new(vec![-1.0; curr_gradient.size]),
                            gradient: None,
                            requires_grad: false,
                        };
                        let right_grad = LazyBuffer::from_operation(TensorOperation::Multiply(
                            Arc::new(Tensor {
                                buffer: curr_gradient.clone(),
                                gradient: None,
                                requires_grad: false,
                            }),
                            Arc::new(neg_one),
                        ));

                        left_tensor.gradient = Some(curr_gradient.clone());
                        right_tensor.gradient = Some(right_grad.clone());

                        queue.push_back((left_tensor, curr_gradient));
                        queue.push_back((right_tensor, right_grad));
                    }
                    TensorOperation::Creation => {
                        // Base case: accumulate gradient
                        if self.gradient.is_none() {
                            self.gradient = Some(curr_gradient);
                        } else {
                            // Add to existing gradient
                            let existing_grad = self.gradient.as_ref().unwrap().clone();
                            self.gradient = Some(LazyBuffer::from_operation(TensorOperation::Add(
                                Arc::new(Tensor {
                                    buffer: existing_grad,
                                    gradient: None,
                                    requires_grad: false,
                                }),
                                Arc::new(Tensor {
                                    buffer: curr_gradient,
                                    gradient: None,
                                    requires_grad: false,
                                }),
                            )));
                        }
                    }
                    _ => todo!("Other operations not yet implemented"),
                }
            }
        }
        all_tensors_involved
            .iter_mut()
            .filter(|t| !t.gradient.is_none())
            .for_each(|t| {
                t.gradient.as_mut().expect("").realize(backend, true);
            });
        println!(
            "ALL TENSORS INVOLVED IN BACKWARD PASS: {:?}",
            all_tensors_involved
        );
    }
    /// Run a training loop with the given number of iterations
    /// This is a helper method that handles setting up caching and running iterations
    pub fn run_training_loop<F>(backend: &dyn Backend, iterations: usize, mut step_fn: F)
    where
        F: FnMut(usize) -> Tensor,
    {
        // Begin training loop to enable buffer caching
        Tensor::begin_training_loop(backend);

        // Run each iteration
        for i in 0..iterations {
            let mut result = step_fn(i);
            result.realize(backend); // Ensure the result is computed

            // For demonstration, we could print progress
            if i % 100 == 0 && i > 0 {
                println!("Completed {} training iterations", i);
            }
        }

        // End training loop to free cached buffers
        Tensor::end_training_loop(backend.name());
    }

    /// Force the freeing of all cached buffers for a backend
    /// This can be useful to reclaim memory if needed outside of a training loop
    pub fn free_buffer_cache(backend: &dyn Backend) {
        crate::training::free_all_cached_buffers(backend);
    }
}

use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("buffer", &self.buffer.get_comp_graph_viz())
            .field("gradient", &self.gradient)
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}
impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Add(
                self_arc.clone(),
                other_arc.clone(),
            )),
            gradient: None,
            requires_grad: requires_grad,
        }
    }
}
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self.clone());
        let other_arc = Arc::new(other.clone());
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Add(
                self_arc.clone(),
                other_arc.clone(),
            )),
            gradient: None,
            requires_grad: requires_grad,
        }
    }
}
impl Sub for Tensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Subtract(
                self_arc.clone(),
                other_arc.clone(),
            )),
            gradient: None,
            requires_grad: requires_grad,
        }
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Multiply(
                self_arc.clone(),
                other_arc.clone(),
            )),
            gradient: None,
            requires_grad: requires_grad,
        }
    }
}
impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self.clone());
        let other_arc = Arc::new(other.clone());
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Multiply(
                self_arc.clone(),
                other_arc.clone(),
            )),
            gradient: None,
            requires_grad: requires_grad,
        }
    }
}
impl Div for Tensor {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let requires_grad = self.requires_grad || other.requires_grad;
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            gradient: None,
            requires_grad: requires_grad,
            buffer: LazyBuffer::from_operation(TensorOperation::Divide(
                self_arc.clone(),
                other_arc.clone(),
            )),
        }
    }
}
#[derive(Debug, Clone)]
pub enum TensorOperation {
    Creation,
    Add(Arc<Tensor>, Arc<Tensor>),
    Subtract(Arc<Tensor>, Arc<Tensor>),
    Multiply(Arc<Tensor>, Arc<Tensor>),
    Divide(Arc<Tensor>, Arc<Tensor>),
}

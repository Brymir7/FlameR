use crate::lazybuffer::{Backend, LazyBuffer, LazyBufferHandle, LazyOp};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
    sync::Mutex,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);
thread_local! {
    static  TENSOR_REGISTRY: RefCell<Vec<Tensor>> = RefCell::new(Vec::new());
    static  OP_CACHE : RefCell<HashMap<(LazyBufferHandle, LazyBufferHandle), TensorId>> = RefCell::new(HashMap::new());
}
thread_local! {
    static TENSOR_ID_COUNTER: RefCell<usize> = RefCell  ::new(0);
}
fn get_next_tensor_id() -> TensorId {
    TENSOR_ID_COUNTER.with_borrow_mut(|c| {
        let id = *c;
        *c += 1;
        TensorId(id)
    })
}
#[derive(Clone, Copy)]
pub struct Tensor {
    pub id: TensorId,
    pub buffer: LazyBufferHandle,
    pub gradient: Option<LazyBufferHandle>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        let id = get_next_tensor_id();
        let t = Tensor {
            id,
            buffer: LazyBuffer::new(id, data),
            gradient: None,
            requires_grad: true,
        };
        TENSOR_REGISTRY.with_borrow_mut(|r| {
            r.push(t.clone());
        });
        t
    }
    pub fn without_grad(data: Vec<f32>) -> Self {
        let id = get_next_tensor_id();
        let t = Tensor {
            id,
            buffer: LazyBuffer::new(id, data),
            gradient: None,
            requires_grad: false,
        };
        TENSOR_REGISTRY.with_borrow_mut(|r| {
            r.push(t.clone());
        });
        t
    }

    fn from_operation(op: LazyOp) -> Self {
        match op {
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                if let Some(id) = OP_CACHE.with_borrow_mut(|c| c.get(&(a, b)).cloned()) {
                    return TENSOR_REGISTRY.with_borrow(|r| r[id.0].clone());
                }
            }
            _ => {}
        }
        let id = get_next_tensor_id();
        let t = Tensor {
            id,
            buffer: LazyBuffer::from_operation(id, op),
            gradient: None,
            requires_grad: true,
        };
        TENSOR_REGISTRY.with_borrow_mut(|r| {
            r.push(t.clone());
        });
        match t.buffer.get_op() {
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                OP_CACHE.with_borrow_mut(|c| {
                    c.insert((a, b), id);
                });
            }
            _ => {}
        }

        t
    }
    pub fn realize(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, false);
    }
    pub fn realize_to_host(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, true);
    }
    pub fn begin_training_loop(backend: &dyn Backend) {}

    pub fn end_training_loop(backend_name: &str) {}
    pub fn apply_backward(&mut self, backend: &dyn Backend, lr: f32) {
        self.realize(backend);
        self.backward(backend);
        let temp_buffer = backend
            .allocate_temporary_buffer(&vec![lr; self.buffer.get_size()], self.buffer.get_size());
        TENSOR_REGISTRY.with_borrow_mut(|r| {
            for tensor in r {
                if tensor.gradient.is_some() {
                    backend.multiply(
                        &tensor
                            .gradient
                            .as_ref()
                            .unwrap()
                            .get_device_handle()
                            .unwrap(),
                        &temp_buffer,
                        &tensor
                            .gradient
                            .as_ref()
                            .unwrap()
                            .get_device_handle()
                            .unwrap(),
                        tensor.buffer.get_size(),
                    );
                    backend.subtract(
                        &tensor.buffer.get_device_handle().unwrap(),
                        &tensor
                            .gradient
                            .as_ref()
                            .unwrap()
                            .get_device_handle()
                            .unwrap(),
                        &tensor.buffer.get_device_handle().unwrap(),
                        tensor.buffer.get_size(),
                    );
                }
            }
        });
    }
    // need to accumulate here otherwise residual gradient will be lost
    pub fn backward(&mut self, backend: &dyn Backend) {
        // currTensor, chainRule gradient
        let mut queue = VecDeque::<(Tensor, LazyBufferHandle)>::new();
        self.gradient = Some(LazyBuffer::without_parent(vec![
            1.0;
            self.buffer.get_size()
        ]));
        queue.push_back((
            self.clone(),
            LazyBuffer::without_parent(vec![1.0; self.buffer.get_size()]),
        ));

        while let Some((curr_tensor, chain_rule_gradient)) = queue.pop_front() {
            if !curr_tensor.requires_grad {}
            match curr_tensor.buffer.get_op() {
                LazyOp::Add(a, b) => {
                    TENSOR_REGISTRY.with_borrow_mut(|r| {
                        let a_tensor = &mut r[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(chain_rule_gradient.clone());
                        queue.push_back((a_tensor.clone(), chain_rule_gradient.clone()));
                    });

                    TENSOR_REGISTRY.with_borrow_mut(|r| {
                        let b_tensor = &mut r[b.get_tensor_id().unwrap().0];
                        b_tensor.gradient = Some(chain_rule_gradient.clone());
                        queue.push_back((b_tensor.clone(), chain_rule_gradient));
                    });
                }
                LazyOp::Subtract(a, b) => {
                    TENSOR_REGISTRY.with_borrow_mut(|r| {
                        let a_tensor = &mut r[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(chain_rule_gradient);
                        queue.push_back((a_tensor.clone(), chain_rule_gradient));
                        let b_tensor = &mut r[b.get_tensor_id().unwrap().0];
                        b_tensor.gradient =
                            Some(LazyBuffer::from_operation_no_parent(LazyOp::Multiply(
                                chain_rule_gradient,
                                LazyBuffer::without_parent(vec![
                                    -1.0;
                                    chain_rule_gradient.get_size()
                                ]),
                            )));
                        queue.push_back((b_tensor.clone(), b_tensor.gradient.clone().unwrap()));
                    });
                }
                LazyOp::Multiply(a, b) => {
                    TENSOR_REGISTRY.with_borrow_mut(|r| {
                        let a_tensor = &mut r[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(LazyBuffer::from_operation_no_parent(
                            LazyOp::Multiply(b, chain_rule_gradient.clone()),
                        ));
                        queue.push_back((a_tensor.clone(), a_tensor.gradient.clone().unwrap()));
                    });
                    TENSOR_REGISTRY.with_borrow_mut(|r| {
                        let b_tensor = &mut r[b.get_tensor_id().unwrap().0];
                        b_tensor.gradient = Some(LazyBuffer::from_operation_no_parent(
                            LazyOp::Multiply(a, chain_rule_gradient),
                        ));
                        queue.push_back((b_tensor.clone(), b_tensor.gradient.clone().unwrap()));
                    });
                }
                _ => {}
            }
        }
        TENSOR_REGISTRY.with_borrow_mut(|r| {
            for tensor in r {
                if tensor.gradient.is_some() {
                    tensor.gradient.as_mut().unwrap().realize(backend, false);
                }
            }
        });
    }
    pub fn run_training_loop<F>(backend: &dyn Backend, iterations: usize, mut step_fn: F)
    where
        F: FnMut(usize) -> Tensor,
    {
        Tensor::begin_training_loop(backend);

        for i in 0..iterations {
            let mut result = step_fn(i);
            result.realize(backend);
            if i % 100 == 0 && i > 0 {
                println!("Completed {} training iterations", i);
            }
        }

        Tensor::end_training_loop(backend.name());
    }
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        TENSOR_REGISTRY.with_borrow(|r| {
            let tensor = &r[self.id.0];
            f.debug_struct("Tensor")
                .field("id", &tensor.id)
                .field("buffer_id", &tensor.buffer)
                .field("gradient", &tensor.gradient)
                .field("requires_grad", &tensor.requires_grad)
                .finish()
        })
    }
}
impl Add for Tensor {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Add(self.buffer, other.buffer))
    }
}
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Add(self.buffer, other.buffer))
    }
}
impl Sub for Tensor {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Subtract(self.buffer, other.buffer))
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Multiply(self.buffer, other.buffer))
    }
}
impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Multiply(self.buffer, other.buffer))
    }
}
impl Div for Tensor {
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        Tensor::from_operation(LazyOp::Divide(self.buffer, other.buffer))
    }
}

use crate::lazybuffer::{Backend, LazyBuffer};
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub buffer: LazyBuffer,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        Tensor {
            buffer: LazyBuffer::new(data),
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
            let result = step_fn(i);

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
            .finish()
    }
}
impl Add for Tensor {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Add(
                self_arc.clone(),
                other_arc.clone(),
            )),
        }
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Subtract(
                self_arc.clone(),
                other_arc.clone(),
            )),
        }
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
            buffer: LazyBuffer::from_operation(TensorOperation::Multiply(
                self_arc.clone(),
                other_arc.clone(),
            )),
        }
    }
}

impl Div for Tensor {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let self_arc = Arc::new(self);
        let other_arc = Arc::new(other);
        Tensor {
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

impl TensorOperation {
    pub fn backward(&self, _grad_output: &Tensor) -> Vec<Option<Tensor>> {
        Vec::new()
    }
}

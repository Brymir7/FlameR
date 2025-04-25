use crate::lazybuffer::{Backend, LazyBuffer, LazyBufferHandle, LazyOp};

#[derive(Clone)]
pub struct Tensor {
    pub buffer: LazyBufferHandle,
    pub gradient: Option<LazyBufferHandle>,
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

    fn from_operation(op: LazyOp) -> Self {
        Tensor {
            buffer: LazyBuffer::from_operation(op),
            gradient: None,
            requires_grad: true,
        }
    }
    pub fn realize(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, false);
    }
    pub fn realize_to_host(&mut self, backend: &dyn Backend) {
        self.buffer.realize(backend, true);
    }
    pub fn begin_training_loop(backend: &dyn Backend) {
        crate::training::begin_training_loop(backend);
    }

    pub fn end_training_loop(backend_name: &str) {
        crate::training::end_training_loop(backend_name);
    }

    pub fn backward(&mut self, backend: &dyn Backend) {
        // currTensor, chainRule gradient
        let mut queue = VecDeque::<(Tensor, LazyBufferHandle)>::new();
        self.gradient = Some(LazyBuffer::new(vec![1.0; self.buffer.get_size()]));
        queue.push_back((
            self.clone(),
            LazyBuffer::new(vec![1.0; self.buffer.get_size()]),
        ));
        let all_tensors = self.buffer.get_compute_graph();
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

    pub fn free_buffer_cache(backend: &dyn Backend) {
        crate::training::free_all_cached_buffers(backend);
    }
}

use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
    sync::{Arc, Mutex, RwLock},
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

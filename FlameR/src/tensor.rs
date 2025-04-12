use crate::lazybuffer::{Device, LazyBuffer};
use std::sync::Arc;

struct TensorHandle(usize);
pub struct Tensor {
    pub buffer: LazyBuffer,
}

impl Tensor {
    pub fn new(data: Vec<f32>) -> Self {
        Tensor {
            buffer: LazyBuffer::new(data),
        }
    }
    pub fn to_device(&mut self, device: Device) {
        self.buffer.to_device(device);
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
    pub fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        Vec::new()
    }
}

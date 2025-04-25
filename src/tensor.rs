use crate::lazybuffer::{Backend, LazyBuffer, LazyBufferHandle, LazyOp};
use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
    sync::Mutex,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);
lazy_static::lazy_static! {
    static ref TENSOR_REGISTRY: Mutex<Vec<Tensor>> = Mutex::new(Vec::new());
}
lazy_static::lazy_static! {
    static ref TENSOR_ID_COUNTER: Mutex<usize> = Mutex::new(0);
}
fn get_next_tensor_id() -> TensorId {
    let mut counter = TENSOR_ID_COUNTER.lock().unwrap();
    let id = *counter;
    *counter += 1;
    TensorId(id)
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
        TENSOR_REGISTRY.lock().unwrap().push(t.clone());
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
        TENSOR_REGISTRY.lock().unwrap().push(t.clone());
        t
    }

    fn from_operation(op: LazyOp) -> Self {
        let id = get_next_tensor_id();
        let t = Tensor {
            id,
            buffer: LazyBuffer::from_operation(id, op),
            gradient: None,
            requires_grad: true,
        };
        TENSOR_REGISTRY.lock().unwrap().push(t.clone());
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
                    {
                        let a_tensor =
                            &mut TENSOR_REGISTRY.lock().unwrap()[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(chain_rule_gradient.clone());
                        queue.push_back((a_tensor.clone(), chain_rule_gradient.clone()));
                    }
                    let b_tensor =
                        &mut TENSOR_REGISTRY.lock().unwrap()[b.get_tensor_id().unwrap().0];
                    b_tensor.gradient = Some(chain_rule_gradient.clone());
                    queue.push_back((b_tensor.clone(), chain_rule_gradient.clone()));
                }
                LazyOp::Subtract(a, b) => {
                    {
                        let a_tensor =
                            &mut TENSOR_REGISTRY.lock().unwrap()[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(chain_rule_gradient);
                        queue.push_back((a_tensor.clone(), chain_rule_gradient));
                    }
                    {
                        let b_tensor =
                            &mut TENSOR_REGISTRY.lock().unwrap()[b.get_tensor_id().unwrap().0];
                        b_tensor.gradient =
                            Some(LazyBuffer::from_operation_no_parent(LazyOp::Multiply(
                                chain_rule_gradient,
                                LazyBuffer::without_parent(vec![
                                    -1.0;
                                    chain_rule_gradient.get_size()
                                ]),
                            )));
                        queue.push_back((b_tensor.clone(), b_tensor.gradient.clone().unwrap()));
                    }
                }
                LazyOp::Multiply(a, b) => {
                    {
                        let a_tensor =
                            &mut TENSOR_REGISTRY.lock().unwrap()[a.get_tensor_id().unwrap().0];
                        a_tensor.gradient = Some(LazyBuffer::from_operation_no_parent(
                            LazyOp::Multiply(b.clone(), chain_rule_gradient.clone()),
                        ));
                        queue.push_back((a_tensor.clone(), a_tensor.gradient.clone().unwrap()));
                    }
                    let b_tensor =
                        &mut TENSOR_REGISTRY.lock().unwrap()[b.get_tensor_id().unwrap().0];
                    b_tensor.gradient = Some(LazyBuffer::from_operation_no_parent(
                        LazyOp::Multiply(a.clone(), chain_rule_gradient),
                    ));
                    queue.push_back((b_tensor.clone(), b_tensor.gradient.clone().unwrap()));
                }
                _ => {}
            }
        }
        for tensor in TENSOR_REGISTRY.lock().unwrap().iter_mut() {
            if tensor.gradient.is_some() {
                tensor.gradient.as_mut().unwrap().realize(backend, true);
                println!(
                    "Tensor ID: {:?}, Gradient: {:?}",
                    tensor.id,
                    tensor.gradient.as_ref().unwrap().get_data()
                );
            }
        }
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
        let tensor = TENSOR_REGISTRY.lock().unwrap()[self.id.0].clone();
        f.debug_struct("Tensor")
            .field("id", &tensor.id)
            .field("buffer", &tensor.buffer.get_data())
            .field("gradient", &tensor.gradient.as_ref().map(|g| g.get_data()))
            .field("requires_grad", &tensor.requires_grad)
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

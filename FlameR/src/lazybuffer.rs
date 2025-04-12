use crate::tensor::{Tensor, TensorOperation};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

static BUFFER_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn get_next_id() -> usize {
    BUFFER_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub trait Backend {
    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn subtract(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn multiply(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn divide(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
}

#[derive(Debug, Default)]
pub struct CPUBackend;

impl Backend for CPUBackend {
    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in CPU add");
        lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect()
    }

    fn subtract(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in CPU subtract");
        lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect()
    }

    fn multiply(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in CPU multiply");
        lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect()
    }

    fn divide(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in CPU divide");
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| if *b == 0.0 { f32::NAN } else { a / b })
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct OpenCLBackend;

impl Backend for OpenCLBackend {
    fn add(&self, _lhs: &[f32], _rhs: &[f32]) -> Vec<f32> {
        unimplemented!("OpenCL add not yet implemented");
    }
    fn subtract(&self, _lhs: &[f32], _rhs: &[f32]) -> Vec<f32> {
        unimplemented!("OpenCL subtract not yet implemented");
    }
    fn multiply(&self, _lhs: &[f32], _rhs: &[f32]) -> Vec<f32> {
        unimplemented!("OpenCL multiply not yet implemented");
    }
    fn divide(&self, _lhs: &[f32], _rhs: &[f32]) -> Vec<f32> {
        unimplemented!("OpenCL divide not yet implemented");
    }
}

#[derive(Debug)]
pub struct LazyBuffer {
    pub(crate) id: usize,
    pub(crate) data: Option<Vec<f32>>,
    pub(crate) operation: TensorOperation,
}

impl LazyBuffer {
    pub fn new(data: Vec<f32>) -> Self {
        LazyBuffer {
            id: get_next_id(),
            data: Some(data),
            operation: TensorOperation::Creation,
        }
    }
    pub fn from_operation(operation: TensorOperation) -> Self {
        LazyBuffer {
            id: get_next_id(),
            data: None,
            operation,
        }
    }

    fn build_graph_string_recursive(
        buffer: &LazyBuffer,
        visited: &mut std::collections::HashSet<usize>,
        graph_string: &mut String,
        indent: usize,
    ) {
        write!(graph_string, "{:indent$}", "", indent = indent * 2)
            .expect("Failed to write indentation");

        writeln!(
            graph_string,
            "Buffer(id={}, op={:?}, data={:?})",
            buffer.id, buffer.operation, buffer.data
        )
        .expect("Failed to write buffer info");

        if !visited.insert(buffer.id) {
            write!(graph_string, "{:indent$}", "", indent = (indent + 1) * 2)
                .expect("Failed to write indentation");
            writeln!(graph_string, "... (already visited)")
                .expect("Failed to write visited marker");
            return;
        }

        match &buffer.operation {
            TensorOperation::Creation => {}
            TensorOperation::Add(left_tensor, right_tensor)
            | TensorOperation::Subtract(left_tensor, right_tensor)
            | TensorOperation::Multiply(left_tensor, right_tensor)
            | TensorOperation::Divide(left_tensor, right_tensor) => {
                Self::build_graph_string_recursive(
                    &left_tensor.buffer,
                    visited,
                    graph_string,
                    indent + 1,
                );
                Self::build_graph_string_recursive(
                    &right_tensor.buffer,
                    visited,
                    graph_string,
                    indent + 1,
                );
            }
        }
    }

    pub fn get_comp_graph_viz(&self) -> String {
        let mut graph_string = String::new();
        let mut visited = std::collections::HashSet::new();

        Self::build_graph_string_recursive(self, &mut visited, &mut graph_string, 0);

        graph_string
    }

    fn schedule_computation_recursive(
        buffer: &LazyBuffer,
        schedule: &mut Vec<usize>,
        visited: &mut HashSet<usize>,
    ) {
        if !visited.insert(buffer.id) {
            return;
        }

        match &buffer.operation {
            TensorOperation::Creation => {}
            TensorOperation::Add(left_tensor, right_tensor)
            | TensorOperation::Subtract(left_tensor, right_tensor)
            | TensorOperation::Multiply(left_tensor, right_tensor)
            | TensorOperation::Divide(left_tensor, right_tensor) => {
                Self::schedule_computation_recursive(&left_tensor.buffer, schedule, visited);
                Self::schedule_computation_recursive(&right_tensor.buffer, schedule, visited);
            }
        }

        schedule.push(buffer.id);
    }

    pub fn get_computation_schedule(&self) -> Vec<usize> {
        let mut schedule = Vec::new();
        let mut visited = HashSet::new();
        Self::schedule_computation_recursive(self, &mut schedule, &mut visited);
        schedule
    }

    fn compute(
        &self,
        backend: &dyn Backend,
        buffer_cache: &mut HashMap<usize, Vec<f32>>,
    ) -> Vec<f32> {
        if let Some(data) = buffer_cache.get(&self.id) {
            return data.clone();
        }

        if let Some(data) = &self.data {
            buffer_cache.insert(self.id, data.clone());
            return data.clone();
        }

        let result = match &self.operation {
            TensorOperation::Creation => {
                panic!(
                    "Attempted to compute a 'Creation' buffer (id={}) with no initial data.",
                    self.id
                );
            }
            TensorOperation::Add(left, right)
            | TensorOperation::Subtract(left, right)
            | TensorOperation::Multiply(left, right)
            | TensorOperation::Divide(left, right) => {
                let left_data = left.buffer.compute(backend, buffer_cache);
                let right_data = right.buffer.compute(backend, buffer_cache);

                match self.operation {
                    TensorOperation::Add(_, _) => backend.add(&left_data, &right_data),
                    TensorOperation::Subtract(_, _) => backend.subtract(&left_data, &right_data),
                    TensorOperation::Multiply(_, _) => backend.multiply(&left_data, &right_data),
                    TensorOperation::Divide(_, _) => backend.divide(&left_data, &right_data),
                    _ => unreachable!(),
                }
            }
        };

        buffer_cache.insert(self.id, result.clone());
        result
    }

    pub fn realize(&mut self, backend: &dyn Backend) -> &[f32] {
        if self.data.is_none() {
            let mut cache = HashMap::new();
            let computed_data = self.compute(backend, &mut cache);
            self.data = Some(computed_data);
        }

        self.data
            .as_ref()
            .expect("Data should be computed and stored by now")
    }
}

use crate::tensor::{Tensor, TensorOperation};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use ocl::{Buffer, Context, Device, Platform, Program, Queue};
use std::sync::{Arc, Mutex};

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

#[derive(Debug)]
pub struct OpenCLBackend {
    context: Context,
    queue: Queue,
    program: Program,
    buffer_cache: Arc<Mutex<HashMap<usize, Buffer<f32>>>>,
}

impl Default for OpenCLBackend {
    fn default() -> Self {
        let platform = Platform::default();
        let device = Device::first(platform).expect("No OpenCL device found");
        
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .expect("Failed to create OpenCL context");
            
        let queue = Queue::new(&context, device, None)
            .expect("Failed to create command queue");
            
        let program_src = r#"
            __kernel void add(__global const float* lhs, 
                             __global const float* rhs,
                             __global float* result,
                             const unsigned int len) {
                const int gid = get_global_id(0);
                if (gid < len) {
                    result[gid] = lhs[gid] + rhs[gid];
                }
            }
            
            __kernel void subtract(__global const float* lhs,
                                  __global const float* rhs,
                                  __global float* result,
                                  const unsigned int len) {
                const int gid = get_global_id(0);
                if (gid < len) {
                    result[gid] = lhs[gid] - rhs[gid];
                }
            }
            
            __kernel void multiply(__global const float* lhs,
                                  __global const float* rhs,
                                  __global float* result,
                                  const unsigned int len) {
                const int gid = get_global_id(0);
                if (gid < len) {
                    result[gid] = lhs[gid] * rhs[gid];
                }
            }
            
            __kernel void divide(__global const float* lhs,
                                __global const float* rhs,
                                __global float* result,
                                const unsigned int len) {
                const int gid = get_global_id(0);
                if (gid < len) {
                    result[gid] = (rhs[gid] == 0.0f) ? NAN : lhs[gid] / rhs[gid];
                }
            }
        "#;
        
        let program = Program::builder()
            .devices(device)
            .src(program_src)
            .build(&context)
            .expect("Failed to build OpenCL program");
            
        OpenCLBackend {
            context,
            queue,
            program,
            buffer_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Backend for OpenCLBackend {
    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, "add")
    }

    fn subtract(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, "subtract")
    }

    fn multiply(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, "multiply")
    }

    fn divide(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, "divide")
    }
}

impl OpenCLBackend {
    fn get_or_create_buffer(&self, id: usize, data: &[f32]) -> Buffer<f32> {
        let mut cache = self.buffer_cache.lock().unwrap();
        
        if let Some(buffer) = cache.get(&id) {
            return buffer.clone();
        }
        
        let buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(data.len())
            .copy_host_slice(data)
            .build()
            .expect("Failed to create OpenCL buffer");
            
        cache.insert(id, buffer.clone());
        buffer
    }
    
    fn execute_binary_op(&self, lhs: &[f32], rhs: &[f32], op_name: &str) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in OpenCL {}", op_name);
        
        let len = lhs.len();
        let lhs_buffer = self.get_or_create_buffer(0, lhs);
        let rhs_buffer = self.get_or_create_buffer(1, rhs);
        
        let result_buffer = Buffer::builder()
            .queue(self.queue.clone())
            .flags(ocl::MemFlags::new().read_write())
            .len(len)
            .build()
            .expect("Failed to create result buffer");
            
        let kernel = ocl::Kernel::builder()
            .program(&self.program)
            .name(op_name)
            .arg(&lhs_buffer)
            .arg(&rhs_buffer)
            .arg(&result_buffer)
            .arg(len as u32)
            .build()
            .expect("Failed to create kernel");
            
        unsafe {
            kernel.enq().expect("Failed to enqueue kernel");
        }
        
        let mut result = vec![0.0f32; len];
        result_buffer
            .read(&mut result)
            .enq()
            .expect("Failed to read result buffer");
            
        result
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

    pub fn keep_on_gpu(&self, backend: &OpenCLBackend) -> bool {
        match &self.data {
            Some(data) => {
                backend.get_or_create_buffer(self.id, data);
                true
            }
            None => false
        }
    }
    
    fn compute_with_gpu_awareness(
        &self,
        backend: &OpenCLBackend,
        buffer_cache: &mut HashMap<usize, Vec<f32>>,
        gpu_buffer_ids: &mut HashSet<usize>,
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
                panic!("Attempted to compute a 'Creation' buffer with no data");
            }
            TensorOperation::Add(left, right) | 
            TensorOperation::Subtract(left, right) |
            TensorOperation::Multiply(left, right) |
            TensorOperation::Divide(left, right) => {
                let left_data = left.buffer.compute_with_gpu_awareness(
                    backend, buffer_cache, gpu_buffer_ids);
                let right_data = right.buffer.compute_with_gpu_awareness(
                    backend, buffer_cache, gpu_buffer_ids);

                gpu_buffer_ids.insert(left.buffer.id);
                gpu_buffer_ids.insert(right.buffer.id);

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
    
    pub fn realize_gpu(&mut self, backend: &OpenCLBackend) -> &[f32] {
        if self.data.is_none() {
            let mut cache = HashMap::new();
            let mut gpu_buffers = HashSet::new();
            let computed_data = self.compute_with_gpu_awareness(backend, &mut cache, &mut gpu_buffers);
            self.data = Some(computed_data);
        }

        self.data
            .as_ref()
            .expect("Data should be computed and stored by now")
    }
}
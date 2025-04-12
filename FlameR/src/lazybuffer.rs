use crate::tensor::{Tensor, TensorOperation};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::{self, GpuFuture};

static BUFFER_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn get_next_id() -> usize {
    BUFFER_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub trait Backend {
    // Arithmetic operations
    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn subtract(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn multiply(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;
    fn divide(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32>;

    // Buffer allocation - new functionality
    fn allocate_buffer(&self, data: &[f32]) -> usize;
    fn free_buffer(&self, id: usize) -> bool;
    fn get_buffer_data(&self, id: usize) -> Option<Vec<f32>>;
}

#[derive(Debug, Default)]
pub struct CPUBackend {
    buffer_cache: Mutex<HashMap<usize, Vec<f32>>>,
}

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

    fn allocate_buffer(&self, data: &[f32]) -> usize {
        let id = get_next_id();
        let mut cache = self.buffer_cache.lock().unwrap();
        cache.insert(id, data.to_vec());
        id
    }

    fn free_buffer(&self, id: usize) -> bool {
        let mut cache = self.buffer_cache.lock().unwrap();
        cache.remove(&id).is_some()
    }

    fn get_buffer_data(&self, id: usize) -> Option<Vec<f32>> {
        let cache = self.buffer_cache.lock().unwrap();
        cache.get(&id).cloned()
    }
}

// Define shader for Vulkano compute operations
mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer LhsData {
            float lhs[];
        };

        layout(set = 0, binding = 1) buffer RhsData {
            float rhs[];
        };

        layout(set = 0, binding = 2) buffer ResultData {
            float result[];
        };

        layout(set = 0, binding = 3) uniform OperationInfo {
            uint op_type;   // 0: add, 1: subtract, 2: multiply, 3: divide
            uint data_len;
        } op_info;

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx < op_info.data_len) {
                float lhs_val = lhs[idx];
                float rhs_val = rhs[idx];
                
                if (op_info.op_type == 0) { // add
                    result[idx] = lhs_val + rhs_val;
                } else if (op_info.op_type == 1) { // subtract
                    result[idx] = lhs_val - rhs_val;
                } else if (op_info.op_type == 2) { // multiply
                    result[idx] = lhs_val * rhs_val;
                } else if (op_info.op_type == 3) { // divide
                    result[idx] = (rhs_val == 0.0) ? 0.0/0.0 : lhs_val / rhs_val;
                }
            }
        }
        "
    }
}

// Structure to hold operation info that will be passed to the shader
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct OperationInfo {
    op_type: u32, // 0: add, 1: subtract, 2: multiply, 3: divide
    data_len: u32,
}

#[derive(Debug)]
pub struct VulkanBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    compute_pipeline: Arc<ComputePipeline>,
    buffer_cache: Arc<Mutex<HashMap<usize, Subbuffer<[f32]>>>>,
}

impl Default for VulkanBackend {
    fn default() -> Self {
        // Create a Vulkan instance
        let library = vulkano::VulkanLibrary::new().expect("Failed to load Vulkan library");
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                ..Default::default()
            },
        )
        .expect("Failed to create Vulkan instance");

        // Select physical device
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
            .next()
            .expect("No physical device available");

        // Find a compute queue family
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
            .expect("Couldn't find a compute queue family") as u32;

        // Create device and queue
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("Failed to create device");

        let queue = queues.next().unwrap();

        // Create memory allocator
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Create command buffer allocator
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        // Create descriptor set allocator
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create compute shader and pipeline
        let shader = compute_shader::load(device.clone()).expect("Failed to create shader");
        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .expect("Failed to create compute pipeline");
        VulkanBackend {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            compute_pipeline,
            buffer_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Backend for VulkanBackend {
    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, 0) // 0 = add
    }

    fn subtract(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, 1) // 1 = subtract
    }

    fn multiply(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, 2) // 2 = multiply
    }

    fn divide(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        self.execute_binary_op(lhs, rhs, 3) // 3 = divide
    }

    fn allocate_buffer(&self, data: &[f32]) -> usize {
        let id = get_next_id();
        let buffer = self.create_gpu_buffer(data);

        let mut cache = self.buffer_cache.lock().unwrap();
        cache.insert(id, buffer);

        id
    }

    fn free_buffer(&self, id: usize) -> bool {
        let mut cache = self.buffer_cache.lock().unwrap();
        cache.remove(&id).is_some()
    }

    fn get_buffer_data(&self, id: usize) -> Option<Vec<f32>> {
        let cache = self.buffer_cache.lock().unwrap();

        if let Some(buffer) = cache.get(&id) {
            // Download data from GPU
            let size = buffer.len();
            // Create a host-visible staging buffer
            let staging_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    ..Default::default()
                },
                (0..size as usize).map(|_| 0.0f32),
            )
            .expect("Failed to create staging buffer");

            // Create a command builder to copy from device buffer to host-visible buffer
            let mut builder = AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("Failed to create command buffer builder");

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    buffer.clone(),
                    staging_buffer.clone(),
                ))
                .expect("Failed to copy buffer");

            let command_buffer = builder.build().expect("Failed to build command buffer");

            // Execute command and wait for it to finish
            let future = sync::now(self.device.clone())
                .then_execute(self.queue.clone(), command_buffer)
                .expect("Failed to execute command buffer")
                .then_signal_fence_and_flush()
                .expect("Failed to signal fence and flush");

            future.wait(None).expect("Failed to wait for future");

            // Read from host-visible buffer
            let content = staging_buffer
                .read()
                .expect("Failed to read staging buffer");
            Some(content.to_vec())
        } else {
            None
        }
    }
}

impl VulkanBackend {
    fn create_gpu_buffer(&self, data: &[f32]) -> Subbuffer<[f32]> {
        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .expect("Failed to create buffer")
    }

    fn execute_binary_op(&self, lhs: &[f32], rhs: &[f32], op_type: u32) -> Vec<f32> {
        assert_eq!(lhs.len(), rhs.len(), "Shape mismatch in Vulkan operation");

        let len = lhs.len();

        // Create input and output buffers
        let lhs_buffer = self.create_gpu_buffer(lhs);
        let rhs_buffer = self.create_gpu_buffer(rhs);
        let result_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..len).map(|_| 0.0f32),
        )
        .expect("Failed to create result buffer");

        // Create operation info uniform buffer
        let op_info = OperationInfo {
            op_type,
            data_len: len as u32,
        };

        let op_info_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            op_info,
        )
        .expect("Failed to create operation info buffer");

        // Create descriptor set
        let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, lhs_buffer.clone()),
                WriteDescriptorSet::buffer(1, rhs_buffer.clone()),
                WriteDescriptorSet::buffer(2, result_buffer.clone()),
                WriteDescriptorSet::buffer(3, op_info_buffer.clone()),
            ],
            [],
        )
        .expect("Failed to create descriptor set");

        // Create command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");

        // Dispatch compute shader
        unsafe {
            builder
                .bind_pipeline_compute(self.compute_pipeline.clone())
                .expect("Failed to bind compute pipeline")
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.compute_pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .expect("Failed to bind descriptor set")
                .dispatch([(len as u32 + 63) / 64, 1, 1])
                .expect("Failed to record dispatch command")
        };

        let command_buffer = builder.build().expect("Failed to build command buffer");

        // Execute command and wait for it to finish
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .expect("Failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush");

        future.wait(None).expect("Failed to wait for future");

        // Create staging buffer to read result
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..len).map(|_| 0.0f32),
        )
        .expect("Failed to create staging buffer");

        // Copy result to staging buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Failed to create command buffer builder");

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                result_buffer,
                staging_buffer.clone(),
            ))
            .expect("Failed to copy buffer");

        let command_buffer = builder.build().expect("Failed to build command buffer");

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .expect("Failed to execute command buffer")
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush");

        future.wait(None).expect("Failed to wait for future");

        // Read result
        let content = staging_buffer
            .read()
            .expect("Failed to read staging buffer");
        content.to_vec()
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

    pub fn keep_on_gpu(&self, backend: &VulkanBackend) -> bool {
        match &self.data {
            Some(data) => {
                backend.allocate_buffer(data);
                true
            }
            None => false,
        }
    }

    fn compute_with_gpu_awareness(
        &self,
        backend: &VulkanBackend,
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
            TensorOperation::Add(left, right)
            | TensorOperation::Subtract(left, right)
            | TensorOperation::Multiply(left, right)
            | TensorOperation::Divide(left, right) => {
                let left_data =
                    left.buffer
                        .compute_with_gpu_awareness(backend, buffer_cache, gpu_buffer_ids);
                let right_data =
                    right
                        .buffer
                        .compute_with_gpu_awareness(backend, buffer_cache, gpu_buffer_ids);

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

    pub fn realize_gpu(&mut self, backend: &VulkanBackend) -> &[f32] {
        if self.data.is_none() {
            let mut cache = HashMap::new();
            let mut gpu_buffers = HashSet::new();
            let computed_data =
                self.compute_with_gpu_awareness(backend, &mut cache, &mut gpu_buffers);
            self.data = Some(computed_data);
        }

        self.data
            .as_ref()
            .expect("Data should be computed and stored by now")
    }
}

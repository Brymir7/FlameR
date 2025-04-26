use ash::vk;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Mutex;

use crate::lazybuffer::{Backend, BufferHandle, LazyBufferHandle};
use crate::vulkan::{Buffer, VulkanBackend as VulkanCore};

pub struct VulkanBackend {
    name: String,
    vulkan: std::rc::Rc<VulkanCore>,
    buffers: Mutex<HashMap<LazyBufferHandle, Buffer>>,
    operation_type: Mutex<HashMap<vk::Pipeline, String>>,
    pipelines: Mutex<HashMap<String, vk::Pipeline>>,
}

impl VulkanBackend {
    pub fn new(app_name: &str) -> Self {
        let vulkan = std::rc::Rc::new(VulkanCore::new(app_name));
        let op_type = HashMap::new();
        let pipelines = HashMap::new();
        VulkanBackend {
            name: "Vulkan".to_string(),
            vulkan,
            buffers: Mutex::new(HashMap::new()),
            operation_type: Mutex::new(op_type),
            pipelines: Mutex::new(pipelines),
        }
    }

    pub fn compile_shader_for_operation(&self, operation: &str) {
        let shader_src = match operation {
            "add" => {
                r#"
                #version 450
                layout(local_size_x = 256) in;
                
                layout(push_constant) uniform PushConstants {
                    uint size;
                } push_constants;
                
                layout(set = 0, binding = 0) buffer TensorA {
                    float data[];
                } tensorA;
                
                layout(set = 0, binding = 1) buffer TensorB {
                    float data[];
                } tensorB;
                
                layout(set = 0, binding = 2) buffer TensorResult {
                    float data[];
                } tensorResult;
                
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (idx < push_constants.size) {
                        tensorResult.data[idx] = tensorA.data[idx] + tensorB.data[idx];
                    }
                }
            "#
            }
            "subtract" => {
                r#"
                #version 450
                layout(local_size_x = 256) in;
                
                layout(push_constant) uniform PushConstants {
                    uint size;
                } push_constants;
                
                layout(set = 0, binding = 0) buffer TensorA {
                    float data[];
                } tensorA;
                
                layout(set = 0, binding = 1) buffer TensorB {
                    float data[];
                } tensorB;
                
                layout(set = 0, binding = 2) buffer TensorResult {
                    float data[];
                } tensorResult;
                
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (idx < push_constants.size) {
                        tensorResult.data[idx] = tensorA.data[idx] - tensorB.data[idx];
                    }
                }
            "#
            }
            "multiply" => {
                r#"
                #version 450
                layout(local_size_x = 256) in;
                
                layout(push_constant) uniform PushConstants {
                    uint size;
                } push_constants;
                
                layout(set = 0, binding = 0) buffer TensorA {
                    float data[];
                } tensorA;
                
                layout(set = 0, binding = 1) buffer TensorB {
                    float data[];
                } tensorB;
                
                layout(set = 0, binding = 2) buffer TensorResult {
                    float data[];
                } tensorResult;
                
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (idx < push_constants.size) {
                        tensorResult.data[idx] = tensorA.data[idx] * tensorB.data[idx];
                    }
                }
            "#
            }
            "divide" => {
                r#"
                #version 450
                layout(local_size_x = 256) in;
                
                layout(push_constant) uniform PushConstants {
                    uint size;
                } push_constants;
                
                layout(set = 0, binding = 0) buffer TensorA {
                    float data[];
                } tensorA;
                
                layout(set = 0, binding = 1) buffer TensorB {
                    float data[];
                } tensorB;
                
                layout(set = 0, binding = 2) buffer TensorResult {
                    float data[];
                } tensorResult;
                
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (idx < push_constants.size) {
                        tensorResult.data[idx] = tensorA.data[idx] / tensorB.data[idx];
                    }
                }
            "#
            }
            _ => panic!("Unknown operation: {}", operation),
        };
        let pipeline = self.vulkan.create_pipeline_for_shader(shader_src);
        let mut pipelines = self.pipelines.lock().unwrap();
        let mut op_types = self.operation_type.lock().unwrap();
        pipelines.insert(operation.to_string(), pipeline);
        op_types.insert(pipeline, operation.to_string());
    }
}

impl Backend for VulkanBackend {
    fn allocate_buffer(&self, lazy_buffer: LazyBufferHandle, size: usize) -> BufferHandle {
        if let Some(buffer) = self.buffers.lock().unwrap().get(&lazy_buffer) {
            return BufferHandle {
                id: lazy_buffer,
                size: buffer.size as usize,
            };
        }
        let buffer_size = (size * size_of::<f32>()) as u64;
        let buffer = self.vulkan.create_gpu_buffer(buffer_size);

        let handle = BufferHandle {
            id: lazy_buffer,
            size,
        };

        self.buffers.lock().unwrap().insert(handle.id, buffer);
        handle
    }
    fn read_buffer(&self, handle: &BufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get(&handle.id) {
            let buffer_size = (handle.size * size_of::<f32>()) as u64;
            let staging_buffer = self.vulkan.create_staging_buffer(buffer_size);

            let fence = self
                .vulkan
                .copy_buffer(buffer, &staging_buffer, buffer_size);
            self.vulkan.wait_for_fence(fence);

            let result = self.vulkan.read_buffer::<f32>(&staging_buffer, handle.size);

            unsafe {
                self.vulkan
                    .device
                    .destroy_buffer(staging_buffer.buffer, None);
                self.vulkan.device.free_memory(staging_buffer.memory, None);
            }

            result
        } else {
            panic!("Buffer with ID {:?} not found", handle.id);
        }
    }
    fn free_buffer(&self, handle: &BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.remove(&handle.id) {
            unsafe {
                self.vulkan.device.destroy_buffer(buffer.buffer, None);
                self.vulkan.device.free_memory(buffer.memory, None);
            }
        }
    }

    fn to_device(&self, data: &[f32], handle: &BufferHandle) {
        let buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get(&handle.id) {
            let staging_buffer = self
                .vulkan
                .create_staging_buffer((data.len() * size_of::<f32>()) as u64);
            self.vulkan.upload_to_buffer(data, &staging_buffer);
            let fence = self.vulkan.copy_buffer(
                &staging_buffer,
                buffer,
                (data.len() * size_of::<f32>()) as u64,
            );
            self.vulkan.wait_for_fence(fence);
            unsafe {
                self.vulkan
                    .device
                    .destroy_buffer(staging_buffer.buffer, None);
                self.vulkan.device.free_memory(staging_buffer.memory, None);
            }
        }
    }

    fn to_host(&self, handle: &BufferHandle, size: usize) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get(&handle.id) {
            let buffer_size = (size * size_of::<f32>()) as u64;
            let staging_buffer = self.vulkan.create_staging_buffer(buffer_size);

            let fence = self
                .vulkan
                .copy_buffer(buffer, &staging_buffer, buffer_size);
            self.vulkan.wait_for_fence(fence);

            let result = self.vulkan.read_buffer::<f32>(&staging_buffer, size);

            unsafe {
                self.vulkan
                    .device
                    .destroy_buffer(staging_buffer.buffer, None);
                self.vulkan.device.free_memory(staging_buffer.memory, None);
            }

            result
        } else {
            panic!("Buffer with ID {:?} not found", handle.id);
        }
    }

    fn add(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let buffers = self.buffers.lock().unwrap();
        if let (Some(buffer_a), Some(buffer_b), Some(result_buffer)) = (
            buffers.get(&a.id),
            buffers.get(&b.id),
            buffers.get(&result.id),
        ) {
            {
                let pipelines = self.pipelines.lock().unwrap();
                if !pipelines.contains_key("add") {
                    drop(pipelines);
                    self.compile_shader_for_operation("add");
                }
            }

            let pipeline = {
                let pipelines = self.pipelines.lock().unwrap();
                *pipelines.get("add").unwrap()
            };

            let fence = self.vulkan.execute_compute_with_pipeline(
                buffer_a,
                buffer_b,
                result_buffer,
                size as u32,
                pipeline,
            );
            self.vulkan.wait_for_fence(fence);
        } else {
            panic!("Buffer not found for addition");
        }
    }

    fn subtract(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let buffers = self.buffers.lock().unwrap();
        if let (Some(buffer_a), Some(buffer_b), Some(result_buffer)) = (
            buffers.get(&a.id),
            buffers.get(&b.id),
            buffers.get(&result.id),
        ) {
            {
                let pipelines = self.pipelines.lock().unwrap();
                if !pipelines.contains_key("subtract") {
                    drop(pipelines);
                    self.compile_shader_for_operation("subtract");
                }
            }

            let pipeline = {
                let pipelines = self.pipelines.lock().unwrap();
                *pipelines.get("subtract").unwrap()
            };

            let fence = self.vulkan.execute_compute_with_pipeline(
                buffer_a,
                buffer_b,
                result_buffer,
                size as u32,
                pipeline,
            );
            self.vulkan.wait_for_fence(fence);
        } else {
            panic!("Buffer not found for subtraction");
        }
    }

    fn multiply(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let buffers = self.buffers.lock().unwrap();
        if let (Some(buffer_a), Some(buffer_b), Some(result_buffer)) = (
            buffers.get(&a.id),
            buffers.get(&b.id),
            buffers.get(&result.id),
        ) {
            {
                let pipelines = self.pipelines.lock().unwrap();
                if !pipelines.contains_key("multiply") {
                    drop(pipelines);
                    self.compile_shader_for_operation("multiply");
                }
            }

            let pipeline = {
                let pipelines = self.pipelines.lock().unwrap();
                *pipelines.get("multiply").unwrap()
            };

            let fence = self.vulkan.execute_compute_with_pipeline(
                buffer_a,
                buffer_b,
                result_buffer,
                size as u32,
                pipeline,
            );
            self.vulkan.wait_for_fence(fence);
        } else {
            panic!("Buffer not found for multiplication");
        }
    }

    fn divide(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let buffers = self.buffers.lock().unwrap();
        if let (Some(buffer_a), Some(buffer_b), Some(result_buffer)) = (
            buffers.get(&a.id),
            buffers.get(&b.id),
            buffers.get(&result.id),
        ) {
            {
                let pipelines = self.pipelines.lock().unwrap();
                if !pipelines.contains_key("divide") {
                    drop(pipelines);
                    self.compile_shader_for_operation("divide");
                }
            }

            let pipeline = {
                let pipelines = self.pipelines.lock().unwrap();
                *pipelines.get("divide").unwrap()
            };

            let fence = self.vulkan.execute_compute_with_pipeline(
                buffer_a,
                buffer_b,
                result_buffer,
                size as u32,
                pipeline,
            );
            self.vulkan.wait_for_fence(fence);
        } else {
            panic!("Buffer not found for division");
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn drop(&self) {
        let buffers = self.buffers.lock().unwrap();
        for buffer in buffers.values() {
            unsafe {
                self.vulkan.device.destroy_buffer(buffer.buffer, None);
                self.vulkan.device.free_memory(buffer.memory, None);
            }
        }
    }
}

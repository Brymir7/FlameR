use ash::{
    Entry,
    vk::{self, Handle},
};
use std::ffi::CString;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

pub struct VulkanBackend {
    pub entry: Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub compute_pipeline: vk::Pipeline,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanBackend {
    pub fn new(app_name: &str) -> Self {
        unsafe {
            let entry = Entry::load().expect("Failed to load Vulkan");

            // Set up application info
            let app_name = CString::new(app_name).unwrap();
            let engine_name = CString::new("No Engine").unwrap();
            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::make_api_version(0, 1, 3, 0));

            // Create instance
            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&[])
                .enabled_extension_names(&[]);

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance");

            // Find compute-capable physical device
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices");

            if physical_devices.is_empty() {
                panic!("No physical devices found");
            }

            // Find a physical device with compute support
            let mut selected_device = None;
            let mut selected_queue_family = 0;

            for physical_device in physical_devices {
                let queue_family_properties =
                    instance.get_physical_device_queue_family_properties(physical_device);

                for (index, properties) in queue_family_properties.iter().enumerate() {
                    if properties.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        selected_device = Some(physical_device);
                        selected_queue_family = index as u32;
                        break;
                    }
                }

                if selected_device.is_some() {
                    break;
                }
            }

            let physical_device = selected_device.expect("No compute-capable device found");

            // Log device name
            let device_properties = instance.get_physical_device_properties(physical_device);
            let device_name = std::ffi::CStr::from_ptr(device_properties.device_name.as_ptr())
                .to_str()
                .unwrap_or("Unknown device")
                .to_owned();
            println!("Using device: {}", device_name);

            // Create logical device
            let queue_priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(selected_queue_family)
                .queue_priorities(&queue_priorities)
                .build();

            let features = vk::PhysicalDeviceFeatures::builder().build();
            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&[])
                .enabled_features(&features);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device");

            // Get queue
            let queue = device.get_device_queue(selected_queue_family, 0);

            // Create command pool
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(selected_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .expect("Failed to create command pool");

            // Get memory properties
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            // Create descriptor set layout
            let bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];

            let descriptor_layout_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

            let descriptor_set_layout = device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .expect("Failed to create descriptor set layout");

            // Create push constant range
            let push_constant_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<u32>() as u32)
                .build();

            // Create pipeline layout
            let set_layouts = [descriptor_set_layout];
            let push_constant_ranges = [push_constant_range];

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout");

            // Compile shader for addition (default operation)
            let shader_spirv = Self::compile_shader(
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

                layout(set = 0, binding = 2) buffer TensorD {
                    float data[];
                } tensorD;

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (idx < push_constants.size) {
                        tensorD.data[idx] = tensorA.data[idx] + tensorB.data[idx];
                    }
                }
                "#,
            );

            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::builder().code(&shader_spirv);

            let shader_module = device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module");

            // Create compute pipeline
            let entry_point = CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_point)
                .build();

            let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage)
                .layout(pipeline_layout)
                .build();

            let compute_pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[compute_pipeline_create_info],
                    None,
                )
                .expect("Failed to create compute pipeline")[0];

            // Create descriptor pool
            let pool_sizes = [vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3)
                .build()];

            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(1);

            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create descriptor pool");

            // Allocate descriptor set
            let layouts_for_allocation = [descriptor_set_layout];

            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts_for_allocation);

            let descriptor_set = device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor set")[0];

            // Clean up shader module as it's no longer needed
            device.destroy_shader_module(shader_module, None);

            VulkanBackend {
                entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index: selected_queue_family,
                command_pool,
                descriptor_set_layout,
                pipeline_layout,
                compute_pipeline,
                descriptor_pool,
                descriptor_set,
                memory_properties,
            }
        }
    }

    pub fn compile_shader(source: &str) -> Vec<u32> {
        let compiler = shaderc::Compiler::new().expect("Failed to create shader compiler");
        let compilation_result = compiler
            .compile_into_spirv(
                source,
                shaderc::ShaderKind::Compute,
                "shader.comp",
                "main",
                None,
            )
            .expect("Failed to compile shader");

        compilation_result.as_binary().to_vec()
    }

    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type");
    }

    pub fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .expect("Failed to create buffer");

            let memory_requirements = self.device.get_buffer_memory_requirements(buffer);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    self.find_memory_type(memory_requirements.memory_type_bits, properties),
                );

            let buffer_memory = self
                .device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory");

            self.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind buffer memory");

            Buffer {
                buffer,
                memory: buffer_memory,
                size,
            }
        }
    }

    pub fn create_gpu_buffer(&self, size: u64) -> Buffer {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    pub fn create_staging_buffer(&self, size: u64) -> Buffer {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    pub fn upload_to_buffer<T: Copy>(&self, data: &[T], buffer: &Buffer) {
        let size_in_bytes = (data.len() * std::mem::size_of::<T>()) as u64;
        assert!(
            size_in_bytes <= buffer.size,
            "Data size exceeds buffer size"
        );

        unsafe {
            let mapped_ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory");

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                size_in_bytes as usize,
            );

            self.device.unmap_memory(buffer.memory);
        }
    }

    pub fn copy_buffer(&self, src_buffer: &Buffer, dst_buffer: &Buffer, size: u64) -> vk::Fence {
        unsafe {
            let command_buffer = self.begin_single_time_command();

            let copy_region = vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(size)
                .build();

            self.device.cmd_copy_buffer(
                command_buffer,
                src_buffer.buffer,
                dst_buffer.buffer,
                &[copy_region],
            );

            self.end_single_time_command(command_buffer)
        }
    }

    pub fn begin_single_time_command(&self) -> vk::CommandBuffer {
        unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = self
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate command buffer")[0];

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin command buffer");

            command_buffer
        }
    }

    pub fn end_single_time_command(&self, command_buffer: vk::CommandBuffer) -> vk::Fence {
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
            let fence_create_info = vk::FenceCreateInfo::builder();
            let fence = self
                .device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create fence");

            let submit_info = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                p_next: std::ptr::null(),
                wait_semaphore_count: 0,
                p_wait_semaphores: std::ptr::null(),
                p_wait_dst_stage_mask: std::ptr::null(),
                command_buffer_count: 1,
                p_command_buffers: &command_buffer,
                signal_semaphore_count: 0,
                p_signal_semaphores: std::ptr::null(),
            };

            let result = self.device.queue_submit(self.queue, &[submit_info], fence);
            match result {
                Ok(_) => {}
                Err(e) => panic!("Failed to submit queue: {:?}", e),
            }
            fence
        }
    }

    pub fn wait_for_fence(&self, fence: vk::Fence) {
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, std::u64::MAX)
                .expect("Failed to wait for fence");
            self.device.destroy_fence(fence, None);
        }
    }

    pub fn read_buffer<T: Copy>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let size_in_bytes = (count * std::mem::size_of::<T>()) as u64;
        assert!(
            size_in_bytes <= buffer.size,
            "Read size exceeds buffer size"
        );

        let mut result = Vec::with_capacity(count);

        unsafe {
            let mapped_ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory") as *const T;

            result.extend_from_slice(std::slice::from_raw_parts(mapped_ptr, count));

            self.device.unmap_memory(buffer.memory);
        }

        result
    }

    pub fn create_pipeline_for_shader(&self, shader_src: &str) -> vk::Pipeline {
        unsafe {
            // Compile shader
            let shader_spirv = Self::compile_shader(shader_src);
            let shader_module_create_info =
                vk::ShaderModuleCreateInfo::builder().code(&shader_spirv);

            let shader_module = self
                .device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create shader module");

            // Create compute pipeline
            let entry_point = CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_point)
                .build();

            let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage)
                .layout(self.pipeline_layout)
                .build();

            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[compute_pipeline_create_info],
                    None,
                )
                .expect("Failed to create compute pipeline")[0];

            // Clean up shader module as it's no longer needed
            self.device.destroy_shader_module(shader_module, None);

            pipeline
        }
    }

    pub fn execute_compute_with_pipeline(
        &self,
        buffer_a: &Buffer,
        buffer_b: &Buffer,
        result_buffer: &Buffer,
        tensor_size: u32,
        pipeline: vk::Pipeline,
    ) -> vk::Fence {
        unsafe {
            // Update descriptor sets for the buffers
            let buffer_infos = [
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer_a.buffer)
                    .offset(0)
                    .range(buffer_a.size)
                    .build(),
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer_b.buffer)
                    .offset(0)
                    .range(buffer_b.size)
                    .build(),
                vk::DescriptorBufferInfo::builder()
                    .buffer(result_buffer.buffer)
                    .offset(0)
                    .range(result_buffer.size)
                    .build(),
            ];

            let write_descriptor_sets = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[0..1])
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[1..2])
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[2..3])
                    .build(),
            ];

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            // Begin command buffer
            let command_buffer = self.begin_single_time_command();

            // Bind pipeline and descriptor set
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // Push constants
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&[tensor_size]),
            );

            // Dispatch compute shader
            let workgroup_size = 256;
            let dispatch_x = (tensor_size + workgroup_size - 1) / workgroup_size;
            self.device.cmd_dispatch(command_buffer, dispatch_x, 1, 1);

            // End and submit command buffer
            self.end_single_time_command(command_buffer)
        }
    }

    pub fn cleanup(&self) {
        unsafe {
            self.device.destroy_pipeline(self.compute_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        self.cleanup();
    }
}

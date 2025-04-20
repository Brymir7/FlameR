use crate::lazybuffer::{Backend, BufferHandle};
use std::collections::HashMap;
use std::sync::Mutex;

pub struct CPUBackend {
    name: String,
    buffers: Mutex<HashMap<usize, Vec<f32>>>,
}

impl CPUBackend {
    pub fn new() -> Self {
        CPUBackend {
            name: "CPU".to_string(),
            buffers: Mutex::new(HashMap::new()),
        }
    }
}

impl Backend for CPUBackend {
    fn allocate_buffer(&self, size: usize) -> BufferHandle {
        let handle = BufferHandle {
            id: crate::lazybuffer::get_next_buffer_id(),
            size,
        };

        // Initialize with zeros
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(handle.id, vec![0.0; size]);

        handle
    }

    fn free_buffer(&self, handle: &BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.remove(&handle.id);
    }

    fn to_device(&self, data: &[f32], handle: &BufferHandle) {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get_mut(&handle.id) {
            buffer.clear();
            buffer.extend_from_slice(data);
        } else {
            let mut new_buffer = Vec::with_capacity(data.len());
            new_buffer.extend_from_slice(data);
            buffers.insert(handle.id, new_buffer);
        }
    }

    fn to_host(&self, handle: &BufferHandle, _size: usize) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get(&handle.id) {
            buffer.clone()
        } else {
            panic!("Buffer with ID {} not found", handle.id);
        }
    }

    fn add(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let mut buffers = self.buffers.lock().unwrap();

        let a_data = buffers.get(&a.id).expect("Buffer A not found");
        let b_data = buffers.get(&b.id).expect("Buffer B not found");

        let mut result_data = Vec::with_capacity(size);
        for i in 0..size {
            result_data.push(a_data[i] + b_data[i]);
        }

        buffers.insert(result.id, result_data);
    }

    fn subtract(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let mut buffers = self.buffers.lock().unwrap();

        let a_data = buffers.get(&a.id).expect("Buffer A not found");
        let b_data = buffers.get(&b.id).expect("Buffer B not found");

        let mut result_data = Vec::with_capacity(size);
        for i in 0..size {
            result_data.push(a_data[i] - b_data[i]);
        }
        buffers.insert(result.id, result_data);
    }

    fn multiply(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let mut buffers = self.buffers.lock().unwrap();

        let a_data = buffers.get(&a.id).expect("Buffer A not found");
        let b_data = buffers.get(&b.id).expect("Buffer B not found");

        let mut result_data = Vec::with_capacity(size);
        for i in 0..size {
            result_data.push(a_data[i] * b_data[i]);
        }

        buffers.insert(result.id, result_data);
    }

    fn divide(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize) {
        let mut buffers = self.buffers.lock().unwrap();

        let a_data = buffers.get(&a.id).expect("Buffer A not found");
        let b_data = buffers.get(&b.id).expect("Buffer B not found");

        let mut result_data = Vec::with_capacity(size);
        for i in 0..size {
            result_data.push(a_data[i] / b_data[i]);
        }
        buffers.insert(result.id, result_data);
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn drop(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.clear();
    }
}

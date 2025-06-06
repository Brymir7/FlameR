use crate::lazybuffer::{Backend, BufferHandle, LAZYBUFFER_HANDLE_NULL, LazyBufferHandle};
use std::collections::HashMap;
use std::sync::Mutex;

pub struct CPUBackend {
    name: String,
    buffers: Mutex<HashMap<LazyBufferHandle, Vec<f32>>>,
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
    fn allocate_buffer(&self, lazy_buffer: LazyBufferHandle, size: usize) -> BufferHandle {
        if let Some(buffer) = self.buffers.lock().unwrap().get(&lazy_buffer) {
            return BufferHandle {
                id: lazy_buffer,
                size: buffer.len(),
            };
        }
        let handle = BufferHandle {
            id: lazy_buffer,
            size,
        };

        // Initialize with zeros
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(handle.id, vec![0.0; size]);

        handle
    }
    fn allocate_temporary_buffer(&self, data: &[f32], size: usize) -> BufferHandle {
        let handle = BufferHandle {
            id: LAZYBUFFER_HANDLE_NULL,
            size,
        };

        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(handle.id, data.to_vec());

        handle
    }
    fn read_buffer(&self, handle: &BufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        if let Some(buffer) = buffers.get(&handle.id) {
            buffer.clone()
        } else {
            panic!("Buffer with ID {:?} not found", handle.id);
        }
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
            panic!("Buffer with ID {:?} not found", handle.id);
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
    fn memset(&self, a: &BufferHandle, b: &BufferHandle, _: usize) {
        let mut buffers = self.buffers.lock().unwrap();
        let b_data = buffers.get(&b.id).expect("Buffer B not found").clone();
        let a_data = buffers.get_mut(&a.id).expect("Buffer A not found");
        a_data.clone_from_slice(&b_data);
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn drop(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.clear();
    }
}

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use crate::tensor::Tensor;

// The Backend trait defines operations that any tensor backend must implement
pub trait Backend {
    fn allocate_buffer(&self, size: usize) -> BufferHandle;
    fn free_buffer(&self, handle: &BufferHandle);
    fn drop(&self);
    fn to_device(&self, data: &[f32], handle: &BufferHandle);
    fn to_host(&self, handle: &BufferHandle, size: usize) -> Vec<f32>;
    fn add(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize);
    fn subtract(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize);
    fn multiply(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize);
    fn divide(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle, size: usize);
    fn name(&self) -> &str;
}

// Struct to represent a buffer on a particular backend
#[derive(Debug, Clone)]
pub struct BufferHandle {
    pub id: LazyBufferHandle,
    pub size: usize,
}

// Buffer data and its computation graph
#[derive(Clone)]
pub struct LazyBuffer {
    pub data: Option<Vec<f32>>,
    pub size: usize,
    pub operation: LazyOp,
    pub gpu_buffer: Option<BufferHandle>,
    pub is_dirty: bool,
    pub id: LazyBufferHandle,
}

// The find_tensor_by_id method will use a global hashmap to track tensors for updating their GPU buffers
lazy_static::lazy_static! {
    static ref LAZYBUFFER_REGISTRY: Mutex<HashMap<LazyBufferHandle, Arc<RwLock<LazyBuffer>>>> = Mutex::new(HashMap::new());
}

impl fmt::Debug for LazyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyBuffer[{:?}] data {:?}", self.id, self.data)
    }
}

// Global buffer ID counter
lazy_static::lazy_static! {
    static ref NEXT_BUFFER_ID: Mutex<usize> = Mutex::new(0);
}

pub fn get_next_buffer_id() -> LazyBufferHandle {
    let mut id = NEXT_BUFFER_ID.lock().unwrap();
    let current = *id;
    *id += 1;
    LazyBufferHandle(current)
}

impl LazyBuffer {
    pub fn new(data: Vec<f32>) -> LazyBufferHandle {
        let size = data.len();
        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            data: Some(data),
            size,
            operation: LazyOp::Creation,
            gpu_buffer: None,
            is_dirty: true,
            id,
        };

        let buffer_arc = Arc::new(RwLock::new(buffer.clone()));
        LAZYBUFFER_REGISTRY.lock().unwrap().insert(id, buffer_arc);

        buffer.id
    }

    pub fn from_operation(op: LazyOp) -> LazyBufferHandle {
        // Determine size based on the operation
        let size = match &op {
            LazyOp::Creation => 0, // This shouldn't happen but is here for completeness
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                let a_size = LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&a)
                    .unwrap()
                    .read()
                    .unwrap()
                    .size;
                let b_size = LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&b)
                    .unwrap()
                    .read()
                    .unwrap()
                    .size;
                if a_size != b_size {
                    panic!("Size mismatch in operation: {} vs {}", a_size, b_size);
                }
                a_size
            }
        };

        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            data: None,
            size,
            operation: op,
            gpu_buffer: None,
            is_dirty: true,
            id,
        };

        // Register this buffer for future updates
        let buffer_arc = Arc::new(RwLock::new(buffer.clone()));
        LAZYBUFFER_REGISTRY.lock().unwrap().insert(id, buffer_arc);

        buffer.id
    }

    // Get visualization of the computation graph for debugging
    pub fn get_comp_graph_viz(&self) -> String {
        match &self.operation {
            LazyOp::Creation => format!("Data[{:?}]", self.id),
            LazyOp::Add(a, b) => format!(
                "({}+{})",
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&a)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz(),
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&b)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz()
            ),
            LazyOp::Subtract(a, b) => format!(
                "({}-{})",
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&a)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz(),
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&b)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz()
            ),
            LazyOp::Multiply(a, b) => format!(
                "({}*{})",
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&a)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz(),
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&b)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz()
            ),
            LazyOp::Divide(a, b) => format!(
                "({}/{})",
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&a)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz(),
                LAZYBUFFER_REGISTRY
                    .lock()
                    .unwrap()
                    .get(&b)
                    .unwrap()
                    .read()
                    .unwrap()
                    .get_comp_graph_viz()
            ),
        }
    }

    // Find all dependencies in the computation graph
    fn collect_dependencies<'a>(
        &'a self,
        deps: &mut HashMap<LazyBufferHandle, &'a LazyBuffer>,
        visited: &mut HashSet<LazyBufferHandle>,
    ) {
        if visited.contains(&self.id) {
            return;
        }

        visited.insert(self.id);

        match &self.operation {
            LazyOp::Creation => {
                deps.insert(self.id, self);
            }
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                deps.insert(self.id, self);
            }
        }
    }

    // Topologically sort buffer operations
    fn topological_sort(&self) -> Vec<LazyBufferHandle> {
        let mut deps = HashMap::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(&mut deps, &mut visited);

        let mut result = Vec::new();
        let mut temp_mark = HashSet::new();
        let mut perm_mark = HashSet::new();

        fn visit(
            node_id: LazyBufferHandle,
            deps: &HashMap<LazyBufferHandle, &LazyBuffer>,
            temp_mark: &mut HashSet<LazyBufferHandle>,
            perm_mark: &mut HashSet<LazyBufferHandle>,
            result: &mut Vec<LazyBufferHandle>,
        ) {
            if temp_mark.contains(&node_id) {
                panic!("Cycle detected in computation graph");
            }

            if !perm_mark.contains(&node_id) {
                temp_mark.insert(node_id);

                let node = deps.get(&node_id).unwrap();
                match &node.operation {
                    LazyOp::Creation => {}
                    LazyOp::Add(a, b)
                    | LazyOp::Subtract(a, b)
                    | LazyOp::Multiply(a, b)
                    | LazyOp::Divide(a, b) => {
                        visit(*a, deps, temp_mark, perm_mark, result);
                        visit(*b, deps, temp_mark, perm_mark, result);
                    }
                }

                temp_mark.remove(&node_id);
                perm_mark.insert(node_id);
                result.push(node_id);
            }
        }

        for &id in deps.keys() {
            if !perm_mark.contains(&id) {
                visit(id, &deps, &mut temp_mark, &mut perm_mark, &mut result);
            }
        }

        result
    }

    pub fn realize(&mut self, backend: &dyn Backend, to_host: bool) {
        if let Some(_) = &self.data {
            return;
        }
        let order = self.topological_sort();

        let mut buffer_handles = HashMap::new();
        let mut deps = HashMap::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(&mut deps, &mut visited);

        for &id in &order {
            let node = deps.get(&id).unwrap();
            // Use the buffer cache for allocations during training
            let handle = crate::training::get_cached_buffer(backend, node.size);
            buffer_handles.insert(id, handle);
        }

        for &id in &order {
            let node = deps.get(&id).unwrap();
            let result_handle = buffer_handles.get(&id).unwrap();

            match &node.operation {
                LazyOp::Creation => {
                    if let Some(data) = &node.data {
                        backend.to_device(data, result_handle);
                    }
                }
                LazyOp::Add(a, b) => {
                    let a_handle = buffer_handles.get(&a).unwrap();
                    let b_handle = buffer_handles.get(&b).unwrap();
                    backend.add(a_handle, b_handle, result_handle, node.size);
                }
                LazyOp::Subtract(a, b) => {
                    let a_handle = buffer_handles.get(&a).unwrap();
                    let b_handle = buffer_handles.get(&b).unwrap();
                    backend.subtract(a_handle, b_handle, result_handle, node.size);
                }
                LazyOp::Multiply(a, b) => {
                    let a_handle = buffer_handles.get(&a).unwrap();
                    let b_handle = buffer_handles.get(&b).unwrap();
                    backend.multiply(a_handle, b_handle, result_handle, node.size);
                }
                LazyOp::Divide(a, b) => {
                    let a_handle = buffer_handles.get(&a).unwrap();
                    let b_handle = buffer_handles.get(&b).unwrap();
                    backend.divide(a_handle, b_handle, result_handle, node.size);
                }
            }
        }
        if (to_host || matches!(backend.name(), "CPU")) && self.is_dirty {
            // if cpu its not expensive
            let result_handle = buffer_handles.get(&self.id).unwrap();
            let result_data = backend.to_host(result_handle, self.size);

            self.data = Some(result_data);
        }
        for (_, handle) in buffer_handles {
            crate::training::return_buffer_to_cache(backend.name(), handle);
        }
    }
}

// need lazy buffer ops instead
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub struct LazyBufferHandle(usize);
#[derive(Clone)]
pub enum LazyOp {
    Creation,
    Add(LazyBufferHandle, LazyBufferHandle),
    Subtract(LazyBufferHandle, LazyBufferHandle),
    Multiply(LazyBufferHandle, LazyBufferHandle),
    Divide(LazyBufferHandle, LazyBufferHandle),
    Clear(LazyBufferHandle),
}

impl LazyBufferHandle {
    pub fn realize(&self, backend: &dyn Backend, to_host: bool) {
        let mut buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let mut buffer = buffer.get_mut(self).unwrap().write().unwrap();
        buffer.realize(backend, to_host);
    }
    pub fn get_comp_graph_viz(&self) -> String {
        let buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let buffer = buffer.get(self).unwrap().read().unwrap();
        buffer.get_comp_graph_viz()
    }
    pub fn get_data(&self) -> Option<Vec<f32>> {
        let buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let buffer = buffer.get(self).unwrap().read().unwrap();
        buffer.data.clone()
    }
    pub fn get_size(&self) -> usize {
        let buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let buffer = buffer.get(self).unwrap().read().unwrap();
        buffer.size
    }
    pub fn get_op(&self) -> LazyOp {
        let buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let buffer = buffer.get(self).unwrap().read().unwrap();
        buffer.operation.clone()
    }
    pub fn clear(&self) {
        let mut buffer = LAZYBUFFER_REGISTRY.lock().unwrap();
        let mut buffer = buffer.get_mut(self).unwrap().write().unwrap();
        buffer.operation = LazyOp::Clear(*self);
    }
}

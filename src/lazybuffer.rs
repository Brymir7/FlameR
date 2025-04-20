use crate::tensor::TensorOperation;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

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
    pub id: usize,
    pub size: usize,
}

// Buffer data and its computation graph
#[derive(Clone)]
pub struct LazyBuffer {
    pub data: Option<Vec<f32>>,
    pub size: usize,
    pub operation: TensorOperation,
    pub gpu_buffer: Option<BufferHandle>,
    pub is_dirty: bool,
    pub id: usize,
}

// The find_tensor_by_id method will use a global hashmap to track tensors for updating their GPU buffers
lazy_static::lazy_static! {
    static ref TENSOR_REGISTRY: Mutex<HashMap<usize, Arc<RwLock<LazyBuffer>>>> = Mutex::new(HashMap::new());
}

impl fmt::Debug for LazyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyBuffer[{}]", self.id)
    }
}

// Global buffer ID counter
lazy_static::lazy_static! {
    static ref NEXT_BUFFER_ID: Mutex<usize> = Mutex::new(0);
}

pub fn get_next_buffer_id() -> usize {
    let mut id = NEXT_BUFFER_ID.lock().unwrap();
    let current = *id;
    *id += 1;
    current
}

impl LazyBuffer {
    pub fn new(data: Vec<f32>) -> Self {
        let size = data.len();
        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            data: Some(data),
            size,
            operation: TensorOperation::Creation,
            gpu_buffer: None,
            is_dirty: true,
            id,
        };

        // Register this buffer for future updates
        let buffer_arc = Arc::new(RwLock::new(buffer.clone()));
        TENSOR_REGISTRY.lock().unwrap().insert(id, buffer_arc);

        buffer
    }

    pub fn from_operation(op: TensorOperation) -> Self {
        // Determine size based on the operation
        let size = match &op {
            TensorOperation::Creation => 0, // This shouldn't happen but is here for completeness
            TensorOperation::Add(a, b)
            | TensorOperation::Subtract(a, b)
            | TensorOperation::Multiply(a, b)
            | TensorOperation::Divide(a, b) => {
                assert_eq!(a.buffer.size, b.buffer.size, "Tensor sizes must match");
                a.buffer.size
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
        TENSOR_REGISTRY.lock().unwrap().insert(id, buffer_arc);

        buffer
    }

    // Get visualization of the computation graph for debugging
    pub fn get_comp_graph_viz(&self) -> String {
        match &self.operation {
            TensorOperation::Creation => format!("Data[{}]", self.id),
            TensorOperation::Add(a, b) => format!(
                "({}+{})",
                a.buffer.get_comp_graph_viz(),
                b.buffer.get_comp_graph_viz()
            ),
            TensorOperation::Subtract(a, b) => format!(
                "({}-{})",
                a.buffer.get_comp_graph_viz(),
                b.buffer.get_comp_graph_viz()
            ),
            TensorOperation::Multiply(a, b) => format!(
                "({}*{})",
                a.buffer.get_comp_graph_viz(),
                b.buffer.get_comp_graph_viz()
            ),
            TensorOperation::Divide(a, b) => format!(
                "({}/{})",
                a.buffer.get_comp_graph_viz(),
                b.buffer.get_comp_graph_viz()
            ),
        }
    }

    // Find all dependencies in the computation graph
    fn collect_dependencies<'a>(
        &'a self,
        deps: &mut HashMap<usize, &'a LazyBuffer>,
        visited: &mut HashSet<usize>,
    ) {
        if visited.contains(&self.id) {
            return;
        }

        visited.insert(self.id);

        match &self.operation {
            TensorOperation::Creation => {
                deps.insert(self.id, self);
            }
            TensorOperation::Add(a, b)
            | TensorOperation::Subtract(a, b)
            | TensorOperation::Multiply(a, b)
            | TensorOperation::Divide(a, b) => {
                a.buffer.collect_dependencies(deps, visited);
                b.buffer.collect_dependencies(deps, visited);
                deps.insert(self.id, self);
            }
        }
    }

    // Topologically sort buffer operations
    fn topological_sort(&self) -> Vec<usize> {
        let mut deps = HashMap::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(&mut deps, &mut visited);

        let mut result = Vec::new();
        let mut temp_mark = HashSet::new();
        let mut perm_mark = HashSet::new();

        fn visit(
            node_id: usize,
            deps: &HashMap<usize, &LazyBuffer>,
            temp_mark: &mut HashSet<usize>,
            perm_mark: &mut HashSet<usize>,
            result: &mut Vec<usize>,
        ) {
            if temp_mark.contains(&node_id) {
                panic!("Cycle detected in computation graph");
            }

            if !perm_mark.contains(&node_id) {
                temp_mark.insert(node_id);

                let node = deps.get(&node_id).unwrap();
                match &node.operation {
                    TensorOperation::Creation => {}
                    TensorOperation::Add(a, b)
                    | TensorOperation::Subtract(a, b)
                    | TensorOperation::Multiply(a, b)
                    | TensorOperation::Divide(a, b) => {
                        visit(a.buffer.id, deps, temp_mark, perm_mark, result);
                        visit(b.buffer.id, deps, temp_mark, perm_mark, result);
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
                TensorOperation::Creation => {
                    if let Some(data) = &node.data {
                        backend.to_device(data, result_handle);
                    }
                }
                TensorOperation::Add(a, b) => {
                    let a_handle = buffer_handles.get(&a.buffer.id).unwrap();
                    let b_handle = buffer_handles.get(&b.buffer.id).unwrap();
                    backend.add(a_handle, b_handle, result_handle, node.size);
                }
                TensorOperation::Subtract(a, b) => {
                    let a_handle = buffer_handles.get(&a.buffer.id).unwrap();
                    let b_handle = buffer_handles.get(&b.buffer.id).unwrap();
                    backend.subtract(a_handle, b_handle, result_handle, node.size);
                }
                TensorOperation::Multiply(a, b) => {
                    let a_handle = buffer_handles.get(&a.buffer.id).unwrap();
                    let b_handle = buffer_handles.get(&b.buffer.id).unwrap();
                    backend.multiply(a_handle, b_handle, result_handle, node.size);
                }
                TensorOperation::Divide(a, b) => {
                    let a_handle = buffer_handles.get(&a.buffer.id).unwrap();
                    let b_handle = buffer_handles.get(&b.buffer.id).unwrap();
                    backend.divide(a_handle, b_handle, result_handle, node.size);
                }
            }
        }
        if (to_host || matches!(backend.name(), "CPU")) && self.is_dirty { // if cpu its not expensive
            let result_handle = buffer_handles.get(&self.id).unwrap();
            let result_data = backend.to_host(result_handle, self.size);
            self.data = Some(result_data);
        }
        // Return used buffers to cache when done
        for (_, handle) in buffer_handles {
            crate::training::return_buffer_to_cache(backend.name(), handle);
        }
    }
}

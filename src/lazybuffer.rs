use core::panic;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Mutex;

use crate::tensor::TensorId;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct LazyBufferHandle(pub usize);

#[derive(Debug, Clone)]
pub enum LazyOp {
    Creation,
    Clear(LazyBufferHandle),
    Add(LazyBufferHandle, LazyBufferHandle),
    Subtract(LazyBufferHandle, LazyBufferHandle),
    Multiply(LazyBufferHandle, LazyBufferHandle),
    Divide(LazyBufferHandle, LazyBufferHandle),
}

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

#[derive(Debug, Clone)]
pub struct BufferHandle {
    pub id: LazyBufferHandle,
    pub size: usize,
}

#[derive(Clone)]
pub struct LazyBuffer {
    pub id: LazyBufferHandle,
    pub parent: Option<TensorId>,
    pub data: Option<Vec<f32>>,
    pub size: usize,
    pub operation: LazyOp,
    pub gpu_buffer: Option<BufferHandle>,
    pub is_dirty: bool,
}

thread_local! {
    pub static LAZYBUFFER_REGISTRY: RefCell<Vec<LazyBuffer>> = RefCell::new(Vec::new());
}

impl fmt::Debug for LazyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyBuffer[{:?}] data {:?}", self.id, self.data)
    }
}

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
    pub fn new(tensor_id: TensorId, data: Vec<f32>) -> LazyBufferHandle {
        let size = data.len();
        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            data: Some(data),
            size,
            operation: LazyOp::Creation,
            gpu_buffer: None,
            is_dirty: true,
            id,
            parent: Some(tensor_id),
        };
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            registry.push(buffer);
        });
        id
    }
    pub fn without_parent(data: Vec<f32>) -> LazyBufferHandle {
        let size = data.len();
        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            data: Some(data),
            size,
            operation: LazyOp::Creation,
            gpu_buffer: None,
            is_dirty: true,
            id,
            parent: None,
        };
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            registry.push(buffer);
        });
        id
    }
    pub fn from_operation(tensor_id: TensorId, op: LazyOp) -> LazyBufferHandle {
        let size = match &op {
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                println!("Operation: {:?}", op);
                println!(
                    "curr reg: {:?}",
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.len())
                );
                let a_size =
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.get(a.0).unwrap().size);
                let b_size =
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.get(b.0).unwrap().size);
                if a_size != b_size {
                    panic!("Size mismatch in operation: {} vs {}", a_size, b_size);
                }
                a_size
            }
            _ => {
                panic!("Unsupported operation for size calculation: {:?}", op);
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
            parent: Some(tensor_id),
        };
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            registry.push(buffer);
        });
        id
    }
    pub fn from_operation_no_parent(op: LazyOp) -> LazyBufferHandle {
        let size = match &op {
            LazyOp::Creation => 0, // This shouldn't happen but is here for completeness
            LazyOp::Clear(a) => {
                let a_size =
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.get(a.0).unwrap().size);
                a_size
            }
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b) => {
                let a_size =
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.get(a.0).unwrap().size);
                let b_size =
                    LAZYBUFFER_REGISTRY.with_borrow(|registry| registry.get(b.0).unwrap().size);
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
            parent: None,
        };

        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| registry.push(buffer));

        id
    }
    pub fn get_comp_graph_viz(&self) -> String {
        match &self.operation {
            LazyOp::Creation => format!("Data[{:?}]", self.id),
            LazyOp::Clear(_) => format!("Clear[{:?}]", self.id),
            LazyOp::Add(a, b) => format!("({}+{})", a.get_comp_graph_viz(), b.get_comp_graph_viz()),
            LazyOp::Subtract(a, b) => {
                format!("({}-{})", a.get_comp_graph_viz(), b.get_comp_graph_viz())
            }
            LazyOp::Multiply(a, b) => {
                format!("({}*{})", a.get_comp_graph_viz(), b.get_comp_graph_viz())
            }
            LazyOp::Divide(a, b) => {
                format!("({}/{})", a.get_comp_graph_viz(), b.get_comp_graph_viz())
            }
        }
    }

    fn collect_dependencies(&self) -> HashMap<LazyBufferHandle, LazyBuffer> {
        let mut deps = HashMap::new();
        let mut visited = HashSet::new();

        fn collect_recursive(
            current_id: LazyBufferHandle,
            deps: &mut HashMap<LazyBufferHandle, LazyBuffer>,
            visited: &mut HashSet<LazyBufferHandle>,
        ) {
            if visited.contains(&current_id) {
                return;
            }

            visited.insert(current_id);

            let current = LAZYBUFFER_REGISTRY
                .with_borrow(|registry| registry.get(current_id.0).unwrap().clone());

            match &current.operation {
                LazyOp::Creation | LazyOp::Clear(_) => {
                    deps.insert(current_id, current);
                }
                LazyOp::Add(a, b)
                | LazyOp::Subtract(a, b)
                | LazyOp::Multiply(a, b)
                | LazyOp::Divide(a, b) => {
                    collect_recursive(*a, deps, visited);
                    collect_recursive(*b, deps, visited);
                    deps.insert(current_id, current);
                }
            }
        }

        collect_recursive(self.id, &mut deps, &mut visited);
        deps
    }

    fn topological_sort(deps: &HashMap<LazyBufferHandle, LazyBuffer>) -> Vec<LazyBufferHandle> {
        let mut result = Vec::new();
        let mut temp_mark = HashSet::new();
        let mut perm_mark = HashSet::new();

        fn visit(
            node_id: LazyBufferHandle,
            deps: &HashMap<LazyBufferHandle, LazyBuffer>,
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
                    LazyOp::Creation | LazyOp::Clear(_) => {}
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
                visit(id, deps, &mut temp_mark, &mut perm_mark, &mut result);
            }
        }

        result
    }

    fn realize_impl(
        &mut self,
        backend: &dyn Backend,
        to_host: bool,
        deps: HashMap<LazyBufferHandle, LazyBuffer>,
    ) {
        if let Some(_) = &self.data {
            return;
        }

        let order = Self::topological_sort(&deps);
        let mut buffer_handles = HashMap::new();

        // Allocate buffers
        for &id in &order {
            let node = deps.get(&id).unwrap();
            let handle = backend.allocate_buffer(node.size);
            buffer_handles.insert(id, handle);
        }

        // Process operations
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
                _ => {
                    panic!("Unsupported operation: {:?}", node.operation);
                }
            }
        }

        if (to_host || matches!(backend.name(), "CPU")) && self.is_dirty {
            let result_handle = buffer_handles.get(&self.id).unwrap();
            let result_data = backend.to_host(result_handle, self.size);
            self.data = Some(result_data);
        }
    }
}

impl LazyBufferHandle {
    pub fn realize(&self, backend: &dyn Backend, to_host: bool) {
        let deps = LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.collect_dependencies()
        });

        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            // Then realize with the collected dependencies
            let buffer = registry.get_mut(self.0).unwrap();
            buffer.realize_impl(backend, to_host, deps);
        });
    }
    pub fn get_comp_graph_viz(&self) -> String {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.get_comp_graph_viz()
        })
    }

    pub fn get_data(&self) -> Option<Vec<f32>> {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.data.clone()
        })
    }
    pub fn get_size(&self) -> usize {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.size
        })
    }
    pub fn get_op(&self) -> LazyOp {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.operation.clone()
        })
    }
    pub fn clear(&self) {
        todo!()
    }
    pub fn get_tensor_id(&self) -> Option<TensorId> {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.parent
        })
    }
}

use core::panic;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::tensor::TensorId;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LazyBufferHandle(pub usize);
pub const LAZYBUFFER_HANDLE_NULL: LazyBufferHandle = LazyBufferHandle(usize::MAX);
#[derive(Debug, Clone, PartialEq)]
pub enum CreationType {
    Random,
    RawData(Box<[f32]>),
    Created,
}

#[derive(Debug, Clone)]
pub enum LazyOp {
    Creation(CreationType),
    Clear(LazyBufferHandle),
    Add(LazyBufferHandle, LazyBufferHandle),
    Subtract(LazyBufferHandle, LazyBufferHandle),
    Multiply(LazyBufferHandle, LazyBufferHandle),
    Divide(LazyBufferHandle, LazyBufferHandle),
}

pub trait Backend {
    fn allocate_buffer(&self, lazy_buffer: LazyBufferHandle, size: usize) -> BufferHandle;
    fn allocate_temporary_buffer(&self, data: &[f32], size: usize) -> BufferHandle;
    fn read_buffer(&self, handle: &BufferHandle) -> Vec<f32>;
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
    pub size: usize,
    pub operation: LazyOp,
    pub device_buffer: Option<BufferHandle>,
}

thread_local! {
    pub static LAZYBUFFER_REGISTRY: RefCell<Vec<LazyBuffer>> = RefCell::new(Vec::new());
}

thread_local! {
    static  NEXT_BUFFER_ID: RefCell<usize> = RefCell::new(0);
}

pub fn get_next_buffer_id() -> LazyBufferHandle {
    let id = NEXT_BUFFER_ID.with_borrow_mut(|id| {
        let current = *id;
        *id += 1;
        current
    });
    LazyBufferHandle(id)
}

impl LazyBuffer {
    pub fn new(tensor_id: TensorId, data: Vec<f32>) -> LazyBufferHandle {
        let size = data.len();
        let id = get_next_buffer_id();
        let buffer = LazyBuffer {
            size,
            operation: LazyOp::Creation(CreationType::RawData(data.into_boxed_slice())),
            device_buffer: None,
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
            size,
            operation: LazyOp::Creation(CreationType::RawData(data.into_boxed_slice())),
            device_buffer: None,
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
            size,
            operation: op,
            device_buffer: None,
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
            LazyOp::Creation(CreationType::RawData(data)) => data.len(),
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
            _ => {
                panic!("Unsupported operation for size calculation: {:?}", op);
            }
        };

        let id = get_next_buffer_id();

        let buffer = LazyBuffer {
            size,
            operation: op,
            device_buffer: None,
            id,
            parent: None,
        };

        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| registry.push(buffer));

        id
    }
    pub fn get_comp_graph_viz(&self) -> String {
        match &self.operation {
            LazyOp::Creation(_) => format!("Data[{:?}]", self.id),
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
                LazyOp::Creation(_) | LazyOp::Clear(_) => {
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
                    LazyOp::Creation(_) | LazyOp::Clear(_) => {}
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
    ) -> HashMap<LazyBufferHandle, BufferHandle> {
        let order = Self::topological_sort(&deps);
        let mut buffer_handles = HashMap::new();

        for &id in &order {
            let node = deps.get(&id).unwrap();
            let handle = backend.allocate_buffer(id, node.size);
            buffer_handles.insert(id, handle);
        }

        for &id in &order {
            let node = deps.get(&id).unwrap();
            let result_handle = buffer_handles.get(&id).unwrap();

            match &node.operation {
                LazyOp::Creation(creation_type) => match creation_type {
                    CreationType::Random => {
                        backend.to_device(&vec![0.0; node.size], result_handle);
                    }
                    CreationType::RawData(data) => {
                        backend.to_device(&data, result_handle);
                    }
                    CreationType::Created => {
                        // buffer has been created already reuse it using handle now
                        continue;
                    }
                },
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
        return buffer_handles;
    }
}

impl LazyBufferHandle {
    pub fn realize(&self, backend: &dyn Backend, to_host: bool) {
        let deps = LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.collect_dependencies()
        });
        let mut buffer_handles: HashMap<LazyBufferHandle, BufferHandle> = HashMap::new();
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            // Then realize with the collected dependencies
            let buffer = registry.get_mut(self.0).unwrap();
            buffer_handles = buffer.realize_impl(backend, to_host, deps);
        });
        for (lazy_buffer, device_handle) in buffer_handles.iter() {
            LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
                let buffer = registry.get_mut(lazy_buffer.0).unwrap();
                buffer.device_buffer = Some(device_handle.clone());
                match &mut buffer.operation {
                    LazyOp::Creation(CreationType::Random)
                    | LazyOp::Creation(CreationType::RawData(_)) => {
                        buffer.operation = LazyOp::Creation(CreationType::Created);
                    }
                    _ => {}
                }
            });
        }
    }
    pub fn get_comp_graph_viz(&self) -> String {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.get_comp_graph_viz()
        })
    }
    pub fn get_data(&self, backend: &dyn Backend) -> Vec<f32> {
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            let buffer = registry.get_mut(self.0).unwrap();
            let device_data = backend.read_buffer(&buffer.device_buffer.as_ref().unwrap());
            device_data
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
    pub fn get_device_handle(&self) -> Option<BufferHandle> {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.device_buffer.clone()
        })
    }
}

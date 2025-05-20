use core::panic;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

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
    Memset(LazyBufferHandle, LazyBufferHandle), // set A to B
}
fn calculate_op_hash(op: &LazyOp) -> Option<usize> {
    let mut hasher = DefaultHasher::new();

    match op {
        LazyOp::Creation(_) => return None, // Don't cache creation ops
        LazyOp::Clear(a) => {
            a.0.hash(&mut hasher);
            1_usize.hash(&mut hasher); // Operation type identifier
        }
        LazyOp::Add(a, b) => {
            // Sort handles to ensure (a+b) and (b+a) hash the same
            let (min, max) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
            min.hash(&mut hasher);
            max.hash(&mut hasher);
            2_usize.hash(&mut hasher);
        }
        LazyOp::Multiply(a, b) => {
            // Sort handles to ensure (a*b) and (b*a) hash the same
            let (min, max) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
            min.hash(&mut hasher);
            max.hash(&mut hasher);
            3_usize.hash(&mut hasher);
        }
        LazyOp::Subtract(a, b) => {
            // Order matters for subtraction
            a.0.hash(&mut hasher);
            b.0.hash(&mut hasher);
            4_usize.hash(&mut hasher);
        }
        LazyOp::Divide(a, b) => {
            // Order matters for division
            a.0.hash(&mut hasher);
            b.0.hash(&mut hasher);
            5_usize.hash(&mut hasher);
        }
        LazyOp::Memset(a, b) => {
            a.0.hash(&mut hasher);
            b.0.hash(&mut hasher);
            6_usize.hash(&mut hasher);
        }
    }

    Some(hasher.finish() as usize)
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
    fn memset(&self, a: &BufferHandle, b: &BufferHandle, size: usize);
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct BufferHandle {
    pub id: LazyBufferHandle,
    pub size: usize,
}

#[derive(Debug, Clone)]
enum LazybufferType {
    Scratch,
    TensorData(TensorId),
}
#[derive(Clone)]
pub struct LazyBuffer {
    pub id: LazyBufferHandle,
    pub kind: LazybufferType,
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
// todo cache here all ::scratch buffers, we can cache them and reuse them cause their data should be immutable
thread_local! {
    static  SCRATCHPAD_CACHE: RefCell<HashMap<usize, LazyBufferHandle>> = RefCell::new(HashMap::new());
}
thread_local! {
    static  SCRATCH_PAD_OP_CACHE: RefCell<HashMap<usize, LazyBufferHandle>> = RefCell::new(HashMap::new());
}
// todo cache all gradient overwrites here
thread_local! {
    static TENSOR_TO_BUFFERS: RefCell<HashMap<TensorId, Vec<LazyBufferHandle>>> = RefCell::new(HashMap::new());
}
pub fn get_next_buffer_id() -> LazyBufferHandle {
    let id = NEXT_BUFFER_ID.with_borrow_mut(|id| {
        let current = *id;
        *id += 1;
        current
    });
    LazyBufferHandle(id)
}
fn calculate_data_hash(data: &[f32]) -> usize {
    let mut hasher = DefaultHasher::new();
    for value in data {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish() as usize
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
            kind: LazybufferType::TensorData(tensor_id),
        };
        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            registry.push(buffer);
        });
        id
    }
    // should be exclusively used for temporary buffers that are not directly linked to any tensor
    pub fn scratch(data: Vec<f32>) -> LazyBufferHandle {
        let size = data.len();
        let data_hash = calculate_data_hash(&data);
        let cached_handle = SCRATCHPAD_CACHE.with_borrow(|cache| cache.get(&data_hash).cloned());

        if let Some(handle) = cached_handle {
            return handle;
        }

        let id = get_next_buffer_id();
        let buffer = LazyBuffer {
            size,
            operation: LazyOp::Creation(CreationType::RawData(data.into_boxed_slice())),
            device_buffer: None,
            id,
            kind: LazybufferType::Scratch,
        };

        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
            registry.push(buffer);
        });

        SCRATCHPAD_CACHE.with_borrow_mut(|cache| {
            cache.insert(data_hash, id);
        });

        id
    }
    pub fn from_tensor_op(tensor_id: TensorId, op: LazyOp) -> LazyBufferHandle {
        let tensor_buffers = TENSOR_TO_BUFFERS.with_borrow(|cache| cache.get(&tensor_id).cloned());
        if let Some(tensor_buffers) = tensor_buffers {
            for buffer_handle in tensor_buffers {
                if let Some(buffer) = LAZYBUFFER_REGISTRY
                    .with_borrow(|registry| registry.get(buffer_handle.0).cloned())
                {
                    match (&buffer.operation, &op) {
                        (LazyOp::Add(a1, b1), LazyOp::Add(a2, b2))
                        | (LazyOp::Subtract(a1, b1), LazyOp::Subtract(a2, b2))
                        | (LazyOp::Multiply(a1, b1), LazyOp::Multiply(a2, b2))
                        | (LazyOp::Divide(a1, b1), LazyOp::Divide(a2, b2)) => {
                            if (a1 == a2 && b1 == b2) {
                                return buffer_handle;
                            }
                        }
                        _ => continue,
                    }
                } else {
                    panic!("Invalid cache");
                }
            }
        }
        let size = match &op {
            LazyOp::Add(a, b)
            | LazyOp::Subtract(a, b)
            | LazyOp::Multiply(a, b)
            | LazyOp::Divide(a, b)
            | LazyOp::Memset(a, b) => {
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
        match &op {
            LazyOp::Memset(a, b) => {
                let buffer = LazyBuffer {
                    size,
                    operation: op.clone(),
                    device_buffer: None,
                    id: *a,
                    kind: LazybufferType::TensorData(tensor_id),
                };
                LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
                    registry[a.0] = buffer;
                });
                return *a;
            }
            _ => {
                let id = get_next_buffer_id();
                let buffer = LazyBuffer {
                    size,
                    operation: op,
                    device_buffer: None,
                    id,
                    kind: LazybufferType::TensorData(tensor_id),
                };

                LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| {
                    registry.push(buffer);
                });
                TENSOR_TO_BUFFERS.with_borrow_mut(|cache| {
                    cache
                        .entry(tensor_id)
                        .and_modify(|buffers| buffers.push(id))
                        .or_insert_with(|| vec![id]);
                });
                return id;
            }
        };
    }
    pub fn scratch_op(op: LazyOp) -> LazyBufferHandle {
        if let Some(op_hash) = calculate_op_hash(&op) {
            if let Some(cached_handle) =
                SCRATCH_PAD_OP_CACHE.with_borrow(|cache| cache.get(&op_hash).cloned())
            {
                return cached_handle;
            }
        }
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
            operation: op.clone(),
            device_buffer: None,
            id,
            kind: LazybufferType::Scratch,
        };

        LAZYBUFFER_REGISTRY.with_borrow_mut(|registry| registry.push(buffer));
        if let Some(op_hash) = calculate_op_hash(&op) {
            SCRATCH_PAD_OP_CACHE.with_borrow_mut(|cache| {
                cache.insert(op_hash, id);
            });
        }
        id
    }
    pub fn get_comp_graph_viz(&self) -> String {
        match &self.operation {
            LazyOp::Memset(a, b) => {
                format!("({}={})", a.get_comp_graph_viz(), b.get_comp_graph_viz())
            }
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
                | LazyOp::Memset(a, b)
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
                    LazyOp::Memset(_, b) => {
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
                LazyOp::Memset(a, b) => {
                    let a_handle = buffer_handles.get(&a).unwrap();
                    let b_handle = buffer_handles.get(&b).unwrap();
                    backend.memset(a_handle, b_handle, node.size);
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
            match buffer.kind {
                LazybufferType::Scratch => return None,
                LazybufferType::TensorData(id) => return Some(id),
            }
        })
    }
    pub fn get_device_handle(&self) -> Option<BufferHandle> {
        LAZYBUFFER_REGISTRY.with_borrow(|registry| {
            let buffer = registry.get(self.0).unwrap();
            buffer.device_buffer.clone()
        })
    }
}

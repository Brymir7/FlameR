# FlameR

A Rust-based neural network learning project focused on backend abstraction (mainly to also run on GPU). This project is the next iteration of my CPU-based neural network project, redesigned to support multiple compute backends through a unified lazy evaluation system.

## Core Concepts

### Lazy Evaluation
Operations are not immediately executed but rather collected into a computation graph via the `LazyBuffer` system. This allows for:
- Deferred execution of operations
- Backend-specific scheduling

### Backend Abstraction
The library uses a trait-based approach for backend implementations:

```rust
pub trait Backend {
    fn allocate_buffer(&self, lazy_buffer: LazyBufferHandle, size: usize) -> BufferHandle;
    fn read_buffer(&self, handle: &BufferHandle) -> Vec<f32>;
    // ... other operations
}
```

Currently (primitively) implemented backends:
- CPU Backend
- Vulkan Backend

### Tensor Operations
The library provides basic tensor operations with automatic differentiation support:
- Element-wise addition, subtraction, multiplication, division
- Gradient computation and backpropagation


## Implementation Details

The system consists of three main components:

1. **LazyBuffer**: Tracks operations and builds the computation graph
2. **Tensor**: Provides the user-facing API and handles automatic differentiation
3. **Backend**: Implements actual computation execution

Operations are collected until explicitly realized through the backend:

```rust
let mut a = Tensor::new(vec![1.0, 2.0, 3.0]);
let mut w = Tensor::new(vec![0.5, 0.5, 0.5]);
let predictions = a * w;  // Operation is recorded but not executed
predictions.realize(&backend);  // Now the computation is performed
```


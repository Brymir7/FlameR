use crate::tensor::{Tensor, TensorOperation};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

static BUFFER_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn get_next_id() -> usize {
    BUFFER_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
pub enum Device {
    CPU,
    OpenCL,
}
impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}
#[derive(Debug)]
pub struct LazyBuffer {
    id: usize,
    data: Option<Vec<f32>>,
    operation: TensorOperation,
    device: Device,
}

impl LazyBuffer {
    pub fn new(data: Vec<f32>) -> Self {
        LazyBuffer {
            id: get_next_id(), // Assign unique ID
            data: Some(data),
            operation: TensorOperation::Creation,
            device: Device::default(),
        }
    }
    pub fn from_operation(operation: TensorOperation) -> Self {
        LazyBuffer {
            id: get_next_id(), // Assign unique ID
            data: None,
            operation,
            device: Device::default(),
        }
    }

    fn build_graph_string_recursive(
        buffer: &LazyBuffer, // Pass the buffer to analyze explicitly
        visited: &mut std::collections::HashSet<usize>, // Track visited nodes by ID
        graph_string: &mut String, // String to build the visualization
        indent: usize,       // Current indentation level
    ) {
        // Indent the line based on depth in the graph
        write!(graph_string, "{:indent$}", "", indent = indent * 2)
            .expect("Failed to write indentation");

        // Print information about the current buffer node
        // Customize the format of TensorOperation printing if needed (e.g., using a name() method)
        writeln!(
            graph_string,
            "Buffer(id={}, op={:?}, data_present={})",
            buffer.id,        // Assumes LazyBuffer has an 'id' field
            buffer.operation, // Relies on Debug trait for TensorOperation
            buffer.data.is_some()
        )
        .expect("Failed to write buffer info");

        // If this node ID has already been visited in this traversal, stop recursion.
        if !visited.insert(buffer.id) {
            write!(graph_string, "{:indent$}", "", indent = (indent + 1) * 2)
                .expect("Failed to write indentation");
            writeln!(graph_string, "... (already visited)")
                .expect("Failed to write visited marker");
            return; // Avoid infinite loops and redundant printing
        }

        // Recurse into the children (inputs) of this operation
        match &buffer.operation {
            TensorOperation::Creation => {
                // Base case: A buffer created directly from data has no further dependencies.
            }
            // Handle binary operations: Add, Subtract, Multiply, Divide
            TensorOperation::Add(left_tensor, right_tensor)
            | TensorOperation::Subtract(left_tensor, right_tensor)
            | TensorOperation::Multiply(left_tensor, right_tensor)
            | TensorOperation::Divide(left_tensor, right_tensor) => {
                // Recursively call for the left operand's buffer
                Self::build_graph_string_recursive(
                    &left_tensor.buffer, // Assumes Tensor has a 'buffer' field
                    visited,
                    graph_string,
                    indent + 1, // Increase indentation for children
                );
                // Recursively call for the right operand's buffer
                Self::build_graph_string_recursive(
                    &right_tensor.buffer, // Assumes Tensor has a 'buffer' field
                    visited,
                    graph_string,
                    indent + 1, // Increase indentation for children
                );
            } // Add cases here for other operations (Unary, etc.) if they exist
              // Example for a hypothetical Unary operation:
              // TensorOperation::ReLU(input_tensor) => {
              //     Self::build_graph_string_recursive(
              //         &input_tensor.buffer,
              //         visited,
              //         graph_string,
              //         indent + 1,
              //     );
              // }
        }
        // Note: We keep the node in 'visited' after returning to prevent
        // re-exploring the same subgraph multiple times if it appears
        // in different branches of the main graph.
    }

    /// Generates a string visualizing the computation graph leading to this buffer.
    ///
    /// This function performs a depth-first traversal of the graph represented
    /// by `TensorOperation` dependencies, starting from the current `LazyBuffer`.
    /// It requires `LazyBuffer` to have a unique `id` field and relies on the
    /// structure of `TensorOperation` and `Tensor`.
    ///
    /// # Returns
    ///
    /// A `String` containing an indented representation of the computation graph.
    ///
    /// # Panics
    ///
    /// Panics if writing to the string fails, which should generally not happen.
    pub fn get_comp_graph_viz(&self) -> String {
        // Required imports (place these at the top of the file or module):
        // use std::collections::HashSet;
        // use std::fmt::Write;

        let mut graph_string = String::new();
        let mut visited = std::collections::HashSet::new(); // Set to track visited buffer IDs

        // Start the recursive traversal from the current buffer ('self')
        Self::build_graph_string_recursive(self, &mut visited, &mut graph_string, 0);

        graph_string // Return the generated visualization string
    }
    pub fn realize(&mut self) {
        // match self.device {
        //     Device::CPU => {
        //         if self.data.is_none() {
        //             let graph = self.get_comp_graph();
        //             println!("Computing graph: {:?}", graph);
        //         }
        //     }
        //     Device::OpenCL => {}
        // }
    }
    pub fn to_device(&mut self, device: Device) {
        self.device = device;
    }
}

pub mod backends;
pub mod lazybuffer;
pub mod tensor;
pub mod vulkan;

use lazybuffer::LazyBuffer;

use crate::backends::{CPUBackend, VulkanBackend};
use crate::tensor::Tensor;
fn main() {
    let vulkan_backend = VulkanBackend::new("Vulkano Test");
    let cpu_backend = CPUBackend::new();
    let mut a = Tensor::new(vec![1.0, 2.0, 3.0]);
    let mut w = Tensor::new(vec![0.5, 0.5, 0.5]);

    let target = Tensor::without_grad(vec![2.0, 4.0, 6.0]);

    for _ in 0..35 {
        let predictions = a * w;
        let mut loss = (target - predictions) * (target - predictions);
        loss.buffer.realize(&vulkan_backend, false);

        loss.apply_backward(&vulkan_backend, 0.1);
        println!("A {:?}", a);
        println!("Loss: {:?}", loss.buffer.get_data(&vulkan_backend));
    }
}

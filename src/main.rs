pub mod backends;
pub mod lazybuffer;
pub mod tensor;
pub mod vulkan;
use lazybuffer::LAZYBUFFER_REGISTRY;

use crate::backends::{CPUBackend, VulkanBackend};
use crate::tensor::Tensor;
fn main() {
    let vulkan_backend = VulkanBackend::new("Vulkano Test");
    let cpu_backend = CPUBackend::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0]);
    let w = Tensor::new(vec![0.5, 0.5, 0.5]);
    let target = Tensor::without_grad(vec![2.0, 4.0, 6.0]);
    for _ in 0..10 {
        let predictions = a * w;
        let mut loss = target - predictions;
        loss.backward(&vulkan_backend);
        println!("a: {:?}", a);
        println!("w: {:?}", w);
        println!("Loss: {:?}", loss);
    }
}

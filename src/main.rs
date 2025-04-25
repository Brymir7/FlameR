pub mod backends;
pub mod lazybuffer;
pub mod tensor;
pub mod vulkan;
use crate::backends::{CPUBackend, VulkanBackend};
use crate::tensor::Tensor;

pub mod training;
fn main() {
    let vulkan_backend = VulkanBackend::new("Vulkano Test");
    let cpu_backend = CPUBackend::new();
    let mut a = Tensor::new(vec![1.0, 2.0, 3.0]);
    let mut w = Tensor::new(vec![0.5, 0.5, 0.5]);
    for _ in 0..10 {
        let target = Tensor::without_grad(vec![2.0, 4.0, 6.0]);
        let predictions = &a * &w;
        let mut loss = target - predictions;
        loss.backward(&vulkan_backend);
        println!("a: {:?}", a);
        // println!("Loss: {:?}", loss);
    }
}

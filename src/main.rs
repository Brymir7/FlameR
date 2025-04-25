pub mod backends;
pub mod lazybuffer;
pub mod tensor;
pub mod vulkan;

use lazybuffer::Backend;

use crate::backends::{CPUBackend, VulkanBackend};
use crate::tensor::Tensor;
use std::sync::Arc;
use std::time::{Duration, Instant};
pub mod training;
#[cfg(test)]
mod tests {
    use crate::backends::{CPUBackend, VulkanBackend};
    use crate::tensor::Tensor;

    #[test]
    fn test_tensor_operations() {
        let vulkan_backend = VulkanBackend::new("Tensor Operation Test");
        let cpu_backend = CPUBackend::new();

        println!("Creating test tensors...");

        let mut data_a = Vec::with_capacity(1000);
        let mut data_b = Vec::with_capacity(1000);
        for i in 0..1000 {
            data_a.push(i as f32);
            data_b.push((i * 2) as f32 + 1.0); // Add 1.0 to avoid division by zero
        }

        let tensor_a = Tensor::new(data_a.clone());
        let tensor_b = Tensor::new(data_b.clone());

        println!("Testing addition operation...");

        let manual_result: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut cpu_addition = tensor_a.clone() + tensor_b.clone();
        cpu_addition.realize(&cpu_backend);
        let cpu_result = cpu_addition.buffer.data.as_ref().unwrap().clone();

        let mut gpu_addition = tensor_a.clone() + tensor_b.clone();
        gpu_addition.realize_to_host(&vulkan_backend);
        let gpu_result = gpu_addition.buffer.data.as_ref().unwrap().clone();

        assert_results_match("addition", &manual_result, &cpu_result, &gpu_result);

        println!("Testing subtraction operation...");

        let manual_result: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mut cpu_subtraction = tensor_a.clone() - tensor_b.clone();
        cpu_subtraction.realize(&cpu_backend);
        let cpu_result = cpu_subtraction.buffer.data.as_ref().unwrap().clone();

        let mut gpu_subtraction = tensor_a.clone() - tensor_b.clone();
        gpu_subtraction.realize_to_host(&vulkan_backend);
        let gpu_result = gpu_subtraction.buffer.data.as_ref().unwrap().clone();

        assert_results_match("subtraction", &manual_result, &cpu_result, &gpu_result);

        println!("Testing multiplication operation...");

        let manual_result: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a * b)
            .collect();

        let mut cpu_multiplication = tensor_a.clone() * tensor_b.clone();
        cpu_multiplication.realize(&cpu_backend);
        let cpu_result = cpu_multiplication.buffer.data.as_ref().unwrap().clone();

        let mut gpu_multiplication = tensor_a.clone() * tensor_b.clone();
        gpu_multiplication.realize_to_host(&vulkan_backend);
        let gpu_result = gpu_multiplication.buffer.data.as_ref().unwrap().clone();

        assert_results_match("multiplication", &manual_result, &cpu_result, &gpu_result);

        println!("Testing division operation...");

        let manual_result: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| a / b)
            .collect();

        let mut cpu_division = tensor_a.clone() / tensor_b.clone();
        cpu_division.realize(&cpu_backend);
        let cpu_result = cpu_division.buffer.data.as_ref().unwrap().clone();

        let mut gpu_division = tensor_a.clone() / tensor_b.clone();
        gpu_division.realize_to_host(&vulkan_backend);
        let gpu_result = gpu_division.buffer.data.as_ref().unwrap().clone();

        assert_results_match("division", &manual_result, &cpu_result, &gpu_result);
    }

    fn assert_results_match(op_name: &str, manual: &[f32], cpu: &[f32], gpu: &[f32]) {
        const EPSILON: f32 = 1e-5;

        assert_eq!(
            manual.len(),
            cpu.len(),
            "{}: Manual and CPU results have different sizes",
            op_name
        );
        assert_eq!(
            manual.len(),
            gpu.len(),
            "{}: Manual and GPU results have different sizes",
            op_name
        );

        for i in 0..manual.len() {
            assert!(
                (manual[i] - cpu[i]).abs() < EPSILON,
                "{} at index {}: Manual value {} differs from CPU value {}",
                op_name,
                i,
                manual[i],
                cpu[i]
            );

            assert!(
                (manual[i] - gpu[i]).abs() < EPSILON,
                "{} at index {}: Manual value {} differs from GPU value {}",
                op_name,
                i,
                manual[i],
                gpu[i]
            );
        }

        println!("✅ {} test passed! All results match.", op_name);
    }
}

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
        a.apply_gradients(&vulkan_backend);
        w.apply_gradients(&vulkan_backend);
        // println!("Loss: {:?}", loss);
    }
}

use lazybuffer::{Backend, CPUBackend, VulkanBackend};
use std::time::Instant;

pub mod lazybuffer;
pub mod tensor;
use tensor::Tensor;

fn benchmark_cpu_backend(size: usize, iterations: usize) {
    let backend = CPUBackend::default();
    println!(
        "Benchmarking CPU with tensor size {} for {} iterations:",
        size, iterations
    );

    // Create input tensors
    let a = Tensor::new(vec![1.0; size]);
    let b = Tensor::new(vec![2.0; size]);

    // Warmup
    let mut result = a + b;
    result.realize(&backend);

    // Benchmark creation + addition
    let start = Instant::now();
    for _ in 0..iterations {
        let t1 = Tensor::new(vec![1.0; size]);
        let t2 = Tensor::new(vec![2.0; size]);
        let mut result = t1 + t2;
        result.realize(&backend);
    }
    let creation_add_time = start.elapsed();

    // Rest of benchmarks...
    // [Add similar code for other operations]

    println!("  Creation + Add: {:?}", creation_add_time);
    // Print other results...
}

fn benchmark_gpu_backend(size: usize, iterations: usize) {
    let backend = VulkanBackend::default();
    println!(
        "Benchmarking GPU with tensor size {} for {} iterations:",
        size, iterations
    );

    // Create input tensors
    let a = Tensor::new(vec![1.0; size]);
    let b = Tensor::new(vec![2.0; size]);

    // Warmup
    let mut result = a + b;
    result.realize_gpu(&backend);

    // Benchmark creation + addition
    let start = Instant::now();
    for _ in 0..iterations {
        let t1 = Tensor::new(vec![1.0; size]);
        let t2 = Tensor::new(vec![2.0; size]);
        let mut result = t1 + t2;
        result.realize_gpu(&backend);
    }
    let creation_add_time = start.elapsed();

    // Rest of benchmarks...
    // [Add similar code for other operations]

    println!("  Creation + Add: {:?}", creation_add_time);
    // Print other results...
}

fn main() {
    let sizes = [1000, 10000, 100000, 100000000];
    let iterations = 10;

    println!("=== FlameR Backend Benchmark ===\n");

    for &size in &sizes {
        benchmark_cpu_backend(size, iterations);
        benchmark_gpu_backend(size, iterations);
        println!("------------------------------");
    }

    // Additional test for very large tensors
    if let Ok(_) = std::env::var("RUN_LARGE_TEST") {
        println!("\n=== Large Tensor Test ===");
        let large_size = 10_000_000;
        benchmark_cpu_backend(large_size, 3);
        benchmark_gpu_backend(large_size, 3);
    }
}

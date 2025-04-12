use lazybuffer::VulkanBackend;

pub mod lazybuffer;
pub mod tensor;

fn main() {
    let backend = VulkanBackend::default();
    
    let t = tensor::Tensor::new(vec![1.0, 2.0, 3.0]);
    let t2 = tensor::Tensor::new(vec![4.0, 5.0, 6.0]);
    
    t.buffer.keep_on_gpu(&backend);
    t2.buffer.keep_on_gpu(&backend);
    
    let mut t3 = t + t2;
    println!("Before computation: {:?}", t3);
    
    t3.realize_gpu(&backend);
    println!("After computation: {:?}", t3);
}
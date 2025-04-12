pub mod lazybuffer;
pub mod tensor;

fn main() {
    let t = tensor::Tensor::new(vec![1.0, 2.0, 3.0]);
    let t2 = tensor::Tensor::new(vec![4.0, 5.0, 6.0]);
    let t3 = t + t2;
    println!("{:?}", t3);
}

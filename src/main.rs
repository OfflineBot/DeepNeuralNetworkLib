use dnn::Activation;
use ndarray::{Array2, array};

fn main() {
    let input_data: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0]];
    let output_data: Array2<f64> = array![[1.0], [0.0]];
    let activation: Activation = Activation::ReLu;

    let output = dnn::train(10_000, 0.001, 10, activation, input_data, output_data).unwrap();

}

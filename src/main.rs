use dnn::{calculate_new_data, ActivationFunction};
use ndarray::{array, Array2};

fn main() {
    let input_data: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0]];
    let output_data: Array2<f64> = array![[1.0], [0.0]];
    let new_data: Array2<f64> = array![[1.0, 1.0], [0.0, 0.0]];
    let activation: ActivationFunction = ActivationFunction::ReLu;

    let output = dnn::train(10_000, 0.001, 10, activation, input_data, output_data).unwrap();
    calculate_new_data(new_data, output.0);
}

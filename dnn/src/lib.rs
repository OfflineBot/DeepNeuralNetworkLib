mod nn;
use nn::{Matrix, Normalize};

use ndarray::{Array1, Array2};

pub enum Activation {
    ReLu,
    Sigmoid,
}

#[allow(unused)]
pub struct TrainingData {
    matrix: Matrix,
    x_mean: Array1<f64>,
    y_mean: Array1<f64>,
    x_std: Array1<f64>,
    y_std: Array1<f64>,
}

#[allow(unused)]
pub fn train(iterations: usize, learning_rate: f32, hidden_layer_size: usize, activation: Activation, input_data: Array2<f64>, output_data: Array2<f64>) -> TrainingData {
    let norm: Normalize = Normalize::normalize(input_data, output_data);
    let matrix: Matrix = Matrix::he_matrix(&norm.x_norm, &norm.y_norm, hidden_layer_size);

    TrainingData {
        matrix: matrix,
        x_mean: norm.x_mean,
        y_mean: norm.y_mean,
        x_std: norm.x_std,
        y_std: norm.y_std,
    }
} 
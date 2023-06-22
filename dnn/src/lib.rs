#![allow(unused_imports)]
mod network;
mod norm;

use std::io;
use network::{Matrix, TrainingData};
use norm::Normalize;

use ndarray::{Array1, Array2};

pub enum Activation {
    ReLu,
    Sigmoid,
}



#[allow(unused)]
pub fn train(iterations: usize, learning_rate: f64, hidden_layer_size: usize, activation: Activation, input_data: Array2<f64>, output_data: Array2<f64>) -> Result<TrainingData, io::Error> {
    let norm: Normalize = Normalize::normalize(input_data, output_data).unwrap();
    let matrix: Matrix = Matrix::he_matrix(&norm.x_norm, &norm.y_norm, hidden_layer_size);
    let new_matrix: Matrix = TrainingData::train_network(iterations, learning_rate, matrix, &norm);

    Ok(TrainingData {
        matrix: new_matrix,
        x_mean: norm.x_mean,
        y_mean: norm.y_mean,
        x_std: norm.x_std,
        y_std: norm.y_std,
    })
} 
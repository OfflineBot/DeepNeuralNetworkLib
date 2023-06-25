use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::{norm::Normalize, ActivationFunction};

struct Activation;
impl Activation {

    #[allow(unused)]
    fn relu(data: &Array2<f64>) -> Array2<f64> {
        let mut new_data: Array2<f64> = data.clone();
        new_data.map_inplace(|x| {
            if *x == 0.0 {
                *x = 0.0;
            }
        });
        new_data
    }

    #[allow(unused)]
    fn deriv_relu(data: &Array2<f64>) -> Array2<f64> {
        let mut new_data: Array2<f64> = data.clone();
        new_data.map_inplace(|x| {
            if *x < 0.0 {
                *x = 0.0;
            } else if *x > 0.0 {
                *x = 1.0;
            }
        });
        new_data
    }

    #[allow(unused)]
    fn sigmoid(data: &Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + data.mapv(|x| (-x).exp()))
    }

    #[allow(unused)]
    fn deriv_sigmoid(data: &Array2<f64>) -> Array2<f64> {
        let sig: Array2<f64> = Activation::sigmoid(&data);
        sig.clone() * (1.0 - sig)
    }
}

#[allow(unused)]
pub struct Matrix {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl Matrix {
    pub fn he_matrix(input: &Array2<f64>, output: &Array2<f64>, hidden_layer: usize) -> Matrix {
        let distribution = Uniform::new(-1.0, 1.0);
        let w1: Array2<f64> = Array2::random((input.shape()[1], hidden_layer), distribution)
            * (2.0 / input.shape()[1] as f64).sqrt();
        let b1: Array1<f64> = Array1::random(hidden_layer, distribution);
        let w2: Array2<f64> = Array2::random((hidden_layer, output.shape()[1]), distribution)
            * (2.0 / hidden_layer as f64).sqrt();
        let b2: Array1<f64> = Array1::random(output.shape()[1], distribution);

        Matrix { w1, b1, w2, b2 }
    }
}

#[allow(unused)]
pub struct Detail {
    pub iteration: usize,
    pub learning_rate: f64,
    pub hidden_layer_size: usize,
    pub activation_function: ActivationFunction,
}

#[allow(unused)]
pub struct TrainingData {
    pub matrix: Matrix,
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub x_std: Array1<f64>,
    pub y_std: Array1<f64>,
}

struct Forward {
    z1: Array2<f64>,
    a1: Array2<f64>,
    z2: Array2<f64>,
}

struct Backward {
    delta1: Array2<f64>,
    delta2: Array2<f64>,
}

impl TrainingData {

    pub fn train_network(
        iterations: usize,
        learning_rate: f64,
        mut matrix: Matrix,
        norm: &Normalize,
        activation: &ActivationFunction,
    ) -> Matrix {
        for i in 0..=iterations {
            if i % 1_000 == 0 {
                println!("{:.2}%", i as f64 / iterations as f64 * 100.0);
            }

            match activation {
                ActivationFunction::ReLu => {
                    let forward: Forward = TrainingData::forward_relu(&norm, &matrix);
                    let backward: Backward = TrainingData::backward_relu(&matrix, &forward, &norm);
                    matrix = TrainingData::update_matrix(matrix, &forward, &backward, &norm, learning_rate);
                }
                ActivationFunction::Sigmoid => {
                    let forward: Forward = TrainingData::forward_sigmoid(&norm, &matrix);
                    let backward: Backward = TrainingData::backward_sigmoid(&matrix, &forward, &norm);
                    matrix = TrainingData::update_matrix(matrix, &forward, &backward, &norm, learning_rate);
                }
            }
            
        }
        matrix
    }

    fn forward_relu(norm: &Normalize, matrix: &Matrix) -> Forward {
        let z1: Array2<f64> = norm.x_norm.dot(&matrix.w1) + &matrix.b1;
        let a1: Array2<f64> = Activation::relu(&z1);
        let z2: Array2<f64> = a1.dot(&matrix.w2) + &matrix.b2;
        Forward { z1, a1, z2 }
    }

    fn backward_relu(matrix: &Matrix, forward: &Forward, norm: &Normalize) -> Backward {
        let delta2: Array2<f64> = &forward.z2 - &norm.y_norm;
        let delta1: Array2<f64> = delta2.dot(&matrix.w2.t()) * Activation::deriv_relu(&forward.z1);
        Backward { delta1, delta2 }
    }

    fn forward_sigmoid(norm: &Normalize, matrix: &Matrix) -> Forward {
        let z1: Array2<f64> = norm.x_norm.dot(&matrix.w1) + &matrix.b1;
        let a1: Array2<f64> = Activation::sigmoid(&z1);
        let z2: Array2<f64> = a1.dot(&matrix.w2) + &matrix.b2;
        Forward { z1, a1, z2 }
    }

    fn backward_sigmoid(matrix: &Matrix, forward: &Forward, norm: &Normalize) -> Backward {
        let delta2: Array2<f64> = &forward.z2 - &norm.y_norm;
        let delta1: Array2<f64> = delta2.dot(&matrix.w2.t()) * Activation::deriv_sigmoid(&forward.z1);
        Backward { delta1, delta2 }
    }
    
    fn update_matrix(
        mut matrix: Matrix,
        forward: &Forward,
        delta: &Backward,
        norm: &Normalize,
        learning_rate: f64,
    ) -> Matrix {
        matrix.w1 = matrix.w1 - learning_rate * norm.x_norm.t().dot(&delta.delta1);
        matrix.b1 = matrix.b1 - learning_rate * delta.delta1.sum_axis(Axis(0));
        matrix.w2 = matrix.w2 - learning_rate * forward.a1.t().dot(&delta.delta2);
        matrix.b2 = matrix.b2 - learning_rate * delta.delta2.sum_axis(Axis(0));
        matrix
    }

    pub fn print_test(data: Array2<f64>, training_data: TrainingData, detail: Detail) {
        let matrix = training_data.matrix;
        let data_norm: Array2<f64> = (data - training_data.x_mean) / training_data.x_std;
        let z1: Array2<f64> = data_norm.dot(&matrix.w1) + &matrix.b1;
        let a1: Array2<f64> = match detail.activation_function {
            ActivationFunction::ReLu => {
                Activation::relu(&z1)
            }
            ActivationFunction::Sigmoid => {
                Activation::relu(&z1)
            }
        };
        let z2: Array2<f64> = a1.dot(&matrix.w2) + &matrix.b2;
        let data_out: Array2<f64> = z2 * training_data.y_std + training_data.y_mean;
        println!("OUT\n{}", data_out);
    }
}

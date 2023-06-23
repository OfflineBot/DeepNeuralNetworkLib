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
    ) -> Matrix {
        for i in 0..=iterations {
            if i % 1_000 == 0 {
                println!("{:.2}%", i as f64 / iterations as f64 * 100.0);
            }
            let forward: Forward = TrainingData::forward(&norm, &matrix);
            let backward: Backward = TrainingData::backward(&matrix, &forward, &norm);
            matrix = TrainingData::update_matrix(matrix, &forward, &backward, &norm, learning_rate);
        }
        matrix
    }
    fn forward(norm: &Normalize, matrix: &Matrix) -> Forward {
        let z1: Array2<f64> = norm.x_norm.dot(&matrix.w1) + &matrix.b1;
        let a1: Array2<f64> = Activation::relu(&z1);
        let z2: Array2<f64> = a1.dot(&matrix.w2) + &matrix.b2;
        Forward { z1, a1, z2 }
    }
    fn backward(matrix: &Matrix, forward: &Forward, norm: &Normalize) -> Backward {
        let delta2: Array2<f64> = &forward.z2 - &norm.y_norm;
        let delta1: Array2<f64> = delta2.dot(&matrix.w1.t()) * Activation::deriv_relu(&forward.z1);
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
}

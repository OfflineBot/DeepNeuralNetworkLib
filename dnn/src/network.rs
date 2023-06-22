use ndarray::{Axis, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use crate::norm::Normalize;

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
        let w1: Array2<f64> = Array2::random((input.shape()[1], hidden_layer), distribution) * (2.0 / input.shape()[1] as f64).sqrt();
        let b1: Array1<f64> = Array1::random(hidden_layer, distribution);
        let w2: Array2<f64> = Array2::random((hidden_layer, output.shape()[1]), distribution) * (2.0 / hidden_layer as f64).sqrt();
        let b2: Array1<f64> = Array1::random(output.shape()[1], distribution);

        Matrix {
            w1,
            b1,
            w2,
            b2,
        }
    }
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
    pub fn train_network(iterations: usize, learning_rate: f64, matrix: Matrix, norm: &Normalize) -> Matrix {
        matrix
    }
    fn forward() -> Forward {

    }
    fn backward() -> Backward {

    }
}
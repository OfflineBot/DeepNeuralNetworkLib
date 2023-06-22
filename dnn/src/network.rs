use ndarray::{Axis, Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

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


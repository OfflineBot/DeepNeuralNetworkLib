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

pub struct Normalize {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub x_std: Array1<f64>,
    pub y_std: Array1<f64>,
    pub x_norm: Array2<f64>,
    pub y_norm: Array2<f64>,
}

impl Normalize {
    pub fn normalize(x: Array2<f64>, y: Array2<f64>) -> Normalize {
        let x_mean: Array1<f64> = x.mean_axis(Axis(0)).unwrap();
        let y_mean: Array1<f64> = y.mean_axis(Axis(0)).unwrap();
        let x_std_pre: Array1<f64> = x.std_axis(Axis(0), 0.0);
        let y_std_pre: Array1<f64> = y.std_axis(Axis(0), 0.0);
        let x_std = x_std_pre.mapv(|mut x| {
            if x == 0.0 {
                x = 1e-10;
            }
            x
        });
        let y_std = y_std_pre.mapv(|mut x| {
            if x == 0.0 {
                x = 1e-10;
            }
            x
        });
        let x_norm: Array2<f64> = (&x - &x_mean) / &x_std;
        let y_norm: Array2<f64> = (&y - &y_mean) / &y_std;

        Normalize {
            x,
            y,
            x_mean,
            y_mean,
            x_std,
            y_std,
            x_norm,
            y_norm,
        }
    }
}
use ndarray::{Array1, Array2, Axis};
use std::io;

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
    pub fn normalize(x: Array2<f64>, y: Array2<f64>) -> Result<Normalize, io::Error> {
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
        Ok(Normalize {
            x,
            y,
            x_mean,
            y_mean,
            x_std,
            y_std,
            x_norm,
            y_norm,
        })
    }
}
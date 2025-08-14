use crate::linalg;

pub fn rot2(theta: f64) -> linalg::Matrix<2, 2> {
    linalg::Matrix {
        data: [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]],
    }
}

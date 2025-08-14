use crate::linalg;

pub fn rot2(theta: f64) -> linalg::Matrix<2, 2> {
    linalg::Matrix {
        data: [[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]],
    }
}

pub fn trot2(theta: f64) -> linalg::Matrix<3, 3> {
    linalg::Matrix {
        data: [
            [theta.cos(), -theta.sin(), 0.0],
            [theta.sin(), theta.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
}

pub fn transl2(x: f64, y: f64) -> linalg::Matrix<3, 3> {
    linalg::Matrix {
        data: [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
    }
}

pub fn se2(tx: f64, ty: f64, theta: f64) -> linalg::Matrix<3, 3> {
    transl2(tx, ty) * trot2(theta)
}

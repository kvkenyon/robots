//! Minimal, idiomatic linear algebra for f64 with const generics.
//! Vector<N> and Matrix<R, C> implement common operations,
//! determinant (LU), inverse (Gauss–Jordan), and solving linear systems.
//!
//! This is NOT optimized for performance; it's clear and safe by default.

use core::fmt;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

#[inline(always)]
fn almost_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
}

/* ----------------------------- Vector<N> ------------------------------ */

#[derive(Clone, Copy, PartialEq)]
pub struct Vector<const N: usize> {
    pub data: [f64; N],
}

impl<const N: usize> Vector<N> {
    pub const fn new(data: [f64; N]) -> Self {
        Self { data }
    }

    pub fn zeros() -> Self {
        Self { data: [0.0; N] }
    }

    pub fn ones() -> Self {
        Self { data: [1.0; N] }
    }

    pub fn unit(i: usize) -> Self {
        let mut v = [0.0; N];
        assert!(i < N);
        v[i] = 1.0;
        Self { data: v }
    }

    pub fn map<F: FnMut(f64) -> f64>(self, mut f: F) -> Self {
        let mut out = self;
        for x in &mut out.data {
            *x = f(*x);
        }
        out
    }

    pub fn zip_map(self, other: Self, mut f: impl FnMut(f64, f64) -> f64) -> Self {
        let mut out = self;
        for i in 0..N {
            out.data[i] = f(self.data[i], other.data[i]);
        }
        out
    }

    pub fn dot(self, other: Self) -> f64 {
        let mut s = 0.0;
        for i in 0..N {
            s += self.data[i] * other.data[i];
        }
        s
    }

    pub fn norm2(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }

    pub fn normalize(self) -> Self {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize zero vector");
        self / n
    }

    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        for i in 0..N {
            if !almost_eq(self.data[i], other.data[i], eps) {
                return false;
            }
        }
        true
    }
}

// 3D-only helpers
impl Vector<3> {
    pub fn cross(self, other: Self) -> Self {
        let [a1, a2, a3] = self.data;
        let [b1, b2, b3] = other.data;
        Self::new([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
    }
}

/* Display/Debug/Index/Iteration */
impl<const N: usize> fmt::Debug for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Vector").field(&self.data).finish()
    }
}
impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..N {
            write!(f, "{}", self.data[i])?;
            if i + 1 != N {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}
impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}
impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}
impl<const N: usize> IntoIterator for Vector<N> {
    type Item = f64;
    type IntoIter = core::array::IntoIter<f64, N>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/* Arithmetic */
impl<const N: usize> Add for Vector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |a, b| a + b)
    }
}
impl<const N: usize> AddAssign for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
    }
}
impl<const N: usize> Sub for Vector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |a, b| a - b)
    }
}
impl<const N: usize> SubAssign for Vector<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.data[i] -= rhs.data[i];
        }
    }
}
impl<const N: usize> Neg for Vector<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}
impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|x| x * rhs)
    }
}
impl<const N: usize> Mul<Vector<N>> for f64 {
    type Output = Vector<N>;
    fn mul(self, rhs: Vector<N>) -> Self::Output {
        rhs * self
    }
}
impl<const N: usize> Div<f64> for Vector<N> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        assert!(rhs != 0.0);
        self.map(|x| x / rhs)
    }
}
impl<const N: usize> DivAssign<f64> for Vector<N> {
    fn div_assign(&mut self, rhs: f64) {
        assert!(rhs != 0.0);
        for x in &mut self.data {
            *x /= rhs;
        }
    }
}
impl<const N: usize> MulAssign<f64> for Vector<N> {
    fn mul_assign(&mut self, rhs: f64) {
        for x in &mut self.data {
            *x *= rhs;
        }
    }
}

/* ---------------------------- Matrix<R, C> ---------------------------- */

#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<const R: usize, const C: usize> {
    pub data: [[f64; C]; R],
}

impl<const R: usize, const C: usize> Matrix<R, C> {
    pub const fn new(data: [[f64; C]; R]) -> Self {
        Self { data }
    }

    pub fn zeros() -> Self {
        Self {
            data: [[0.0; C]; R],
        }
    }

    pub fn ones() -> Self {
        Self {
            data: [[1.0; C]; R],
        }
    }

    pub fn from_rows(rows: [Vector<C>; R]) -> Self {
        let mut m = [[0.0; C]; R];
        for r in 0..R {
            m[r] = rows[r].data;
        }
        Self { data: m }
    }

    pub fn row(&self, r: usize) -> Vector<C> {
        Vector::new(self.data[r])
    }

    pub fn col(&self, c: usize) -> Vector<R> {
        let mut v = [0.0; R];
        for r in 0..R {
            v[r] = self.data[r][c];
        }
        Vector::new(v)
    }

    pub fn set_row(&mut self, r: usize, v: Vector<C>) {
        self.data[r] = v.data;
    }

    pub fn set_col(&mut self, c: usize, v: Vector<R>) {
        for r in 0..R {
            self.data[r][c] = v.data[r];
        }
    }

    pub fn map<F: FnMut(f64) -> f64>(self, mut f: F) -> Self {
        let mut out = self;
        for r in 0..R {
            for c in 0..C {
                out.data[r][c] = f(out.data[r][c]);
            }
        }
        out
    }

    pub fn transpose(self) -> Matrix<C, R> {
        let mut m = [[0.0; R]; C];
        for r in 0..R {
            for c in 0..C {
                m[c][r] = self.data[r][c];
            }
        }
        Matrix { data: m }
    }

    pub fn approx_eq(&self, other: &Self, eps: f64) -> bool {
        for r in 0..R {
            for c in 0..C {
                if !almost_eq(self.data[r][c], other.data[r][c], eps) {
                    return false;
                }
            }
        }
        true
    }
}

/* Square-matrix extras */
impl<const N: usize> Matrix<N, N> {
    pub fn identity() -> Self {
        let mut m = [[0.0; N]; N];
        for i in 0..N {
            m[i][i] = 1.0;
        }
        Self { data: m }
    }

    /// LU decomposition with partial pivoting (Doolittle).
    /// Returns (LU, pivots, parity_sign). LU combines L (unit diag) and U.
    fn lu_decompose(mut self) -> (Self, [usize; N], f64) {
        let mut piv = [0usize; N];
        for i in 0..N {
            piv[i] = i;
        }
        let mut sign = 1.0;

        for k in 0..N {
            // pivot search
            let mut p = k;
            let mut max_val = self.data[k][k].abs();
            for r in (k + 1)..N {
                let val = self.data[r][k].abs();
                if val > max_val {
                    max_val = val;
                    p = r;
                }
            }
            assert!(max_val > 0.0, "singular matrix in LU decomposition");

            if p != k {
                self.data.swap(p, k);
                piv.swap(p, k);
                sign = -sign;
            }

            // elimination
            for i in (k + 1)..N {
                self.data[i][k] /= self.data[k][k];
                let f = self.data[i][k];
                for j in (k + 1)..N {
                    self.data[i][j] -= f * self.data[k][j];
                }
            }
        }
        (self, piv, sign)
    }

    pub fn det(self) -> f64 {
        let (lu, _, sign) = self.lu_decompose();
        let mut prod = sign;
        for i in 0..N {
            prod *= lu.data[i][i];
        }
        prod
    }

    /// Inverse via Gauss–Jordan with partial pivoting.
    pub fn inverse(mut self) -> Self {
        let mut inv = Self::identity();

        for i in 0..N {
            // find pivot
            let mut pivot_row = i;
            let mut max_val = self.data[i][i].abs();
            for r in (i + 1)..N {
                let val = self.data[r][i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = r;
                }
            }
            assert!(max_val > 0.0, "singular matrix in inversion");

            if pivot_row != i {
                self.data.swap(pivot_row, i);
                inv.data.swap(pivot_row, i);
            }

            // scale pivot row
            let pivot = self.data[i][i];
            for c in 0..N {
                self.data[i][c] /= pivot;
                inv.data[i][c] /= pivot;
            }

            // eliminate other rows
            for r in 0..N {
                if r == i {
                    continue;
                }
                let f = self.data[r][i];
                if f != 0.0 {
                    for c in 0..N {
                        self.data[r][c] -= f * self.data[i][c];
                        inv.data[r][c] -= f * inv.data[i][c];
                    }
                }
            }
        }

        inv
    }

    /// Solve A x = b using forward/back substitution on LU.
    pub fn solve(self, b: Vector<N>) -> Vector<N> {
        let (lu, piv, _) = self.lu_decompose();

        // Apply pivot to b -> Pb
        let mut pb = [0.0; N];
        for i in 0..N {
            pb[i] = b.data[piv[i]];
        }

        // Forward solve L y = P b (L has unit diag)
        let mut y = [0.0; N];
        for i in 0..N {
            let mut s = pb[i];
            for j in 0..i {
                s -= lu.data[i][j] * y[j];
            }
            y[i] = s;
        }

        // Backward solve U x = y
        let mut x = [0.0; N];
        for i in (0..N).rev() {
            let mut s = y[i];
            for j in (i + 1)..N {
                s -= lu.data[i][j] * x[j];
            }
            x[i] = s / lu.data[i][i];
        }

        Vector::new(x)
    }
}

/* Display/Index */
impl<const R: usize, const C: usize> fmt::Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix").field("data", &self.data).finish()
    }
}
impl<const R: usize, const C: usize> fmt::Display for Matrix<R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..R {
            write!(f, "|")?;
            for c in 0..C {
                write!(f, "{}", self.data[r][c])?;
                if c + 1 != C {
                    write!(f, "\t")?;
                }
            }
            writeln!(f, "|")?;
        }
        Ok(())
    }
}
impl<const R: usize, const C: usize> Index<usize> for Matrix<R, C> {
    type Output = [f64; C];
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}
impl<const R: usize, const C: usize> IndexMut<usize> for Matrix<R, C> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

/* Arithmetic */
impl<const R: usize, const C: usize> Add for Matrix<R, C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut m = self;
        for r in 0..R {
            for c in 0..C {
                m.data[r][c] += rhs.data[r][c];
            }
        }
        m
    }
}
impl<const R: usize, const C: usize> AddAssign for Matrix<R, C> {
    fn add_assign(&mut self, rhs: Self) {
        for r in 0..R {
            for c in 0..C {
                self.data[r][c] += rhs.data[r][c];
            }
        }
    }
}
impl<const R: usize, const C: usize> Sub for Matrix<R, C> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut m = self;
        for r in 0..R {
            for c in 0..C {
                m.data[r][c] -= rhs.data[r][c];
            }
        }
        m
    }
}
impl<const R: usize, const C: usize> SubAssign for Matrix<R, C> {
    fn sub_assign(&mut self, rhs: Self) {
        for r in 0..R {
            for c in 0..C {
                self.data[r][c] -= rhs.data[r][c];
            }
        }
    }
}
impl<const R: usize, const C: usize> Neg for Matrix<R, C> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}
impl<const R: usize, const C: usize> Mul<f64> for Matrix<R, C> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        self.map(|x| x * rhs)
    }
}
impl<const R: usize, const C: usize> MulAssign<f64> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: f64) {
        for r in 0..R {
            for c in 0..C {
                self.data[r][c] *= rhs;
            }
        }
    }
}
impl<const R: usize, const C: usize> Div<f64> for Matrix<R, C> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        assert!(rhs != 0.0);
        self.map(|x| x / rhs)
    }
}
impl<const R: usize, const C: usize> DivAssign<f64> for Matrix<R, C> {
    fn div_assign(&mut self, rhs: f64) {
        assert!(rhs != 0.0);
        for r in 0..R {
            for c in 0..C {
                self.data[r][c] /= rhs;
            }
        }
    }
}

/* Matrix × Matrix and Matrix × Vector */
impl<const R: usize, const K: usize, const C: usize> Mul<Matrix<K, C>> for Matrix<R, K> {
    type Output = Matrix<R, C>;
    fn mul(self, rhs: Matrix<K, C>) -> Self::Output {
        let mut out = Matrix::<R, C>::zeros();
        for r in 0..R {
            for c in 0..C {
                let mut s = 0.0;
                for k in 0..K {
                    s += self.data[r][k] * rhs.data[k][c];
                }
                out.data[r][c] = s;
            }
        }
        out
    }
}
impl<const R: usize, const C: usize> Mul<Vector<C>> for Matrix<R, C> {
    type Output = Vector<R>;
    fn mul(self, rhs: Vector<C>) -> Self::Output {
        let mut out = [0.0; R];
        for r in 0..R {
            let mut s = 0.0;
            for c in 0..C {
                s += self.data[r][c] * rhs.data[c];
            }
            out[r] = s;
        }
        Vector::new(out)
    }
}

/* ------------------------------- Tests -------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_ops() {
        let a = Vector::<3>::new([1.0, 2.0, 3.0]);
        let b = Vector::<3>::new([3.0, 2.0, 1.0]);
        assert_eq!((a + b).data, [4.0, 4.0, 4.0]);
        assert_eq!((a - b).data, [-2.0, 0.0, 2.0]);
        assert!((2.0 * a).approx_eq(Vector::new([2.0, 4.0, 6.0]), 1e-12));
        assert!(a.dot(b) - 10.0 < 1e-12);
        let c = a.cross(b);
        assert!(c.approx_eq(Vector::new([-4.0, 8.0, -4.0]), 1e-12));
        let n = Vector::<3>::new([3.0, 0.0, 4.0]);
        assert!(almost_eq(n.norm(), 5.0, 1e-12));
    }

    #[test]
    fn matrix_basic() {
        let a = Matrix::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Matrix::<2, 2>::new([[5.0, 6.0], [7.0, 8.0]]);
        let c = a + b;
        assert_eq!(c.data, [[6.0, 8.0], [10.0, 12.0]]);
        let t = a.transpose();
        assert_eq!(t.data, [[1.0, 3.0], [2.0, 4.0]]);
    }

    #[test]
    fn matmul_and_mv() {
        let a = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0]]);
        let b = Matrix::<3, 2>::new([[1.0, 2.0], [0.0, 1.0], [4.0, 0.0]]);
        let c = a * b;
        assert_eq!(c.data, [[13.0, 4.0], [16.0, 1.0]]);

        let v = Vector::<3>::new([1.0, 2.0, 3.0]);
        let r = a * v;
        assert_eq!(r.data, [14.0, 14.0]);
    }

    #[test]
    fn det_inverse_solve() {
        let a = Matrix::<3, 3>::new([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]);
        let det = a.det();
        assert!(almost_eq(det, 4.0, 1e-12));

        let inv = a.inverse();
        let id = a * inv;
        assert!(id.approx_eq(&Matrix::<3, 3>::identity(), 1e-9));

        let b = Vector::<3>::new([1.0, 0.0, 1.0]);
        let x = a.solve(b);
        // Expected solution ~ [1.0, 1.0, 1.0]
        assert!(x.approx_eq(Vector::new([1.0, 1.0, 1.0]), 1e-9));
    }
}

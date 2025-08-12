use robots::linalg::{Matrix, Vector};

fn main() {
    let v = Vector::<3>::ones();
    println!("v={v}");

    let m = Matrix::<5, 5>::ones();
    println!("{m}");
}

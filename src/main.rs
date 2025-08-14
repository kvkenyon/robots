use robots::motion;

fn main() {
    let rot = motion::rot2(0.3);
    println!("{}", rot);
    println!("det(R)={}", rot.det());
    println!("det(RxR)={}", (rot * rot).det());
}

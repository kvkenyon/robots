use robots::motion;

fn main() {
    let rot = motion::rot2(0.3);
    println!("{}", rot);
    println!("det(R)={}", rot.det());
    println!("det(RxR)={}", (rot * rot).det());

    let degrees: f64 = 30.0;

    let trot = motion::trot2(degrees.to_radians());

    println!("{}", trot);

    let transl = motion::transl2(1.0, 2.0);

    println!("{}", transl);

    let res = transl * trot;
    println!("{}", res);

    let t1 = motion::se2(1.0, 2.0, 30.0f64.to_radians());
    println!("{}", t1);
}

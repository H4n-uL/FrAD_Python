use std::f64::consts::PI;

fn dct(input: Vec<f64>) -> Vec<f64> {
    let slen = input.len();
    let mut output = vec![0.0; slen];

    for i in 0..slen {
        let mut sum = 0.0;
        for j in 0..slen {
            let angle = (2 * j + 1) as f64 * i as f64 * PI / (2 * slen) as f64;
            sum += input[j] * angle.cos();
        }
        output[i] = sum * 2.0;
    }
    return output
}

fn idct(input: Vec<f64>) -> Vec<f64> {
    let slen = input.len();
    let mut output = vec![0.0; slen];

    for i in 0..slen {
        let mut sum = input[0] / 2.0;
        for j in 1..slen {
            let angle = j as f64 * (2 * i + 1) as f64 * PI / (2 * slen) as f64;
            sum += input[j] * angle.cos();
        }
        output[i] = sum / slen as f64;
    }
    return output;
}

fn main() {
    let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
    let output = dct(input.clone());
    let rec = idct(output.clone());

    let output: Vec<f32> = output.iter().map(|&x| x as f32).collect();
    println!("{:?}", input);
    println!("{:?}", output);
    println!("{:?}", rec);
}
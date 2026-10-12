use only_brain::{ActivationFunction::BinaryStep, Perceptron};

fn main() {
    let mut perceptron: Perceptron<2> = Perceptron::new(BinaryStep);

    perceptron.set_weights([1.0, 1.0]);
    perceptron.set_bias(0.1);
    println!("{}", perceptron);

    let input = [2.0, -1.0];
    let output = perceptron.feed_forward(&input);

    println!("Perceptron output for {:?}: {}", input, output);

    let input = [2.0, -1.5];
    let output = perceptron.feed_forward(&input);

    println!("Perceptron output for {:?}: {}", input, output);
}

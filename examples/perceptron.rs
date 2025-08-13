use only_brain::{ActivationFunction::BinaryStep, Perceptron, bvector};

fn main() {
    let mut perceptron: Perceptron<2> = Perceptron::new(BinaryStep);

    let weights = bvector![1.0, 1.0];
    perceptron.set_weights(weights);
    perceptron.set_bias(0.1);

    println!("{}", perceptron);

    let input = bvector![2.0, -1.0];
    let output = perceptron.feed_forward(&input);

    println!("Perceptron output for {:?}: {}", input, output);

    let input = bvector![2.0, -1.5];
    let output = perceptron.feed_forward(&input);

    println!("Perceptron output for {:?}: {}", input, output);
}

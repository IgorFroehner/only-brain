use no_brain::NeuralNetwork;

fn main() {
    let nn = NeuralNetwork::new(&vec![2, 3, 1]);

    let input = vec![0.5, 0.2];
    let output = nn.feed_forward(&input);

    println!("{:?}", output);
}

use no_brain::NeuralNetwork;

fn main() {
    let nn = NeuralNetwork::new(&vec![2, 3, 1]);
    nn.print();
}

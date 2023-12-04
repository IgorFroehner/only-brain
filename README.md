# Only Brain

A very simple Neural Network library built in Rust with the objective to allow
the user to create, manipulate and train a neural network directly. The user has
direct access to weights and biases of the network, allowing they to manipulate
the NN as wanted.

## Usage

```rust
use no_brain::NeuralNetwork;
use nalgebra::dmatrix;
use nalgebra::dvector;

fn main() {
    let mut nn = NeuralNetwork::new(&vec![2, 2, 1]);

    nn.set_layer_weights(1, dmatrix![0.1, 0.2;
                                     0.3, 0.4]);
    nn.set_layer_biases(1, dvector![0.1, 0.2]);

    nn.set_layer_weights(2, dmatrix![0.9, 0.8]);
    nn.set_layer_biases(2, dvector![0.1]);

    let input = vec![0.5, 0.2];
    let output = nn.feed_forward(&input);

    println!("{:?}", output);
}
```

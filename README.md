# Only Brain

A very simple Neural Network library built in Rust with the objective to allow
the user to create, manipulate and train a neural network directly. The user has
direct access to weights and biases of the network, allowing they to manipulate
the NN as wanted.

## Usage

The input and output widths are part of the network's type, so feeding a wrongly
sized input is a compile error. The hidden layers are given at construction time.

```rust
use only_brain::NeuralNetwork;
use nalgebra::dmatrix;
use nalgebra::dvector;

fn main() {
    // A 2 -> 2 -> 1 network.
    let mut nn = NeuralNetwork::<2, 1>::new(&[2]);

    nn.set_layer_weights(1, dmatrix![0.1, 0.2;
                                     0.3, 0.4]);
    nn.set_layer_biases(1, dvector![0.1, 0.2]);

    nn.set_layer_weights(2, dmatrix![0.9, 0.8]);
    nn.set_layer_biases(2, dvector![0.1]);

    let output = nn.feed_forward(&[0.5, 0.2]);

    println!("{:?}", output);
}
```

Saving and loading a model checks the stored shape against the type you ask for:

```rust
use only_brain::{dump_model, load_model, NeuralNetwork};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let nn = NeuralNetwork::<2, 1>::new(&[2]);
dump_model(&nn, "model.bin")?;

let loaded: NeuralNetwork<2, 1> = load_model("model.bin")?;
# Ok(())
# }
```

## TODO

- [ ] Abstract better the DVector and DMatrix from the user's perspective;
    - [x] Perceptron
    - [ ] Neural Network
- [ ] Make it possible to load polars dataframes/datasets;
- [x] Make the types know the size of the NeuralNetwork/inputs and outputs neurons;
    - Input and output widths are compile-time checked; hidden layers stay dynamic.
- [ ] Fix and improve model dump and load;
    - [x] Typed errors, buffered IO, and shape validation on load
    - [ ] Versioned format header and bincode 2 migration


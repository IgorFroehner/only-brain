# Only Brain

A very simple Neural Network library built in Rust with the objective to allow
the user to create, manipulate and train a neural network directly. The user has
direct access to weights and biases of the network, allowing them to manipulate
the NN as wanted: "without the learning part", so you can bring your own.

## Usage

The input and output widths are part of the network's type, so feeding a wrongly
sized input is a compile error. The hidden layers are given at construction time.
Weights and biases are plain arrays, slices or `Vec`s, so you don't need nalgebra
to use the network.

```rust
use only_brain::NeuralNetwork;

fn main() {
    // A 2 -> 2 -> 1 network.
    let mut nn = NeuralNetwork::<2, 1>::new(&[2]);

    // One row per neuron, one weight per neuron of the previous layer.
    nn.set_layer_weights(1, &[[0.1, 0.2],
                              [0.3, 0.4]]);
    nn.set_layer_biases(1, &[0.1, 0.2]);

    nn.set_layer_weights(2, &[[0.9, 0.8]]);
    nn.set_layer_biases(2, &[0.1]);

    let output = nn.feed_forward(&[0.5, 0.2]);

    println!("{:?}", output);
}
```

### Activation per layer

Every layer has its own activation function, so hidden layers and the output layer
can differ. The available ones are `Sigmoid` (the default), `Tanh`, `ReLU`,
`BinaryStep` and `Identity`.

```rust
use only_brain::{ActivationFunction, NeuralNetwork};

let mut nn = NeuralNetwork::<10, 3>::new(&[8]);
nn.set_activation_function(ActivationFunction::ReLU); // every layer
nn.set_output_activation(ActivationFunction::Tanh);   // then just the output
```

### Evolving networks: the flat parameter view

Genetic algorithms and other black-box searches work on a flat list of numbers.
A network converts to and from one in a documented, stable order: layer by layer,
each layer's weights row by row (one row per neuron), then that layer's biases.

```rust
use only_brain::NeuralNetwork;

// The genome length for a 10 -> 8 -> 3 network.
let genes = NeuralNetwork::<10, 3>::parameter_count_for(&[8]); // 115

// In a fitness function: a genome becomes a network.
let genome = vec![0.0; genes];
let nn = NeuralNetwork::<10, 3>::from_parameters(&[8], &genome);

// And a network becomes a genome.
assert_eq!(nn.parameters(), genome);
```

`set_parameters` updates an existing network in place, keeping its activation
functions. `NeuralNetwork` is `Send + Sync`, so fitness can be scored in parallel.
See [`examples/xor_evolution.rs`](examples/xor_evolution.rs) for a complete,
seeded neuroevolution run.

### Saving and loading

Saving and loading a model checks the stored shape against the type you ask for:

```rust,no_run
use only_brain::{dump_model, load_model, NeuralNetwork};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let nn = NeuralNetwork::<2, 1>::new(&[2]);
dump_model(&nn, "model.bin")?;

let loaded: NeuralNetwork<2, 1> = load_model("model.bin")?;
# Ok(())
# }
```

Models are stored in a small, versioned binary format (documented in `src/io.rs`)
that does not depend on any serialization library, so saved models keep loading as
the library evolves. Files written by 0.1 and 0.2 still load. `write_model` and
`read_model` do the same with any `Write` or `Read`, such as a `Vec<u8>`.

`NeuralNetwork` also implements serde's `Serialize` and `Deserialize` through a
plain form (`{"layers": [{"activation", "weights", "biases"}]}`), so it can be
embedded in your own types and formats, such as JSON. Deserializing validates the
shape too.

## Examples

| Example | Shows |
|---|---|
| `neural_network` | Building a network by hand and running it. |
| `dump_load` | Saving and loading a model. |
| `xor_evolution` | Evolving a network with a genetic algorithm through the flat parameter view. |
| `perceptron` | A single perceptron. |
| `perceptron_iris` | Training a perceptron on the Iris dataset with the perceptron rule. |

Run one with `cargo run --example xor_evolution`.

## TODO

- [x] Abstract better the DVector and DMatrix from the user's perspective;
    - [x] Perceptron
    - [x] Neural Network
- [x] Make the types know the size of the NeuralNetwork/inputs and outputs neurons;
    - Input and output widths are compile-time checked; hidden layers stay dynamic.
- [x] Fix and improve model dump and load;
    - [x] Typed errors, buffered IO, and shape validation on load
    - [x] Versioned format header, without bincode
- [x] Flat parameter view, for genetic algorithms
- [x] Activation function per layer

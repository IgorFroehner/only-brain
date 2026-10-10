# Only Brain

[![crates.io](https://img.shields.io/crates/v/only-brain.svg)](https://crates.io/crates/only-brain)
[![docs.rs](https://docs.rs/only-brain/badge.svg)](https://docs.rs/only-brain)
[![CI](https://github.com/IgorFroehner/only-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/IgorFroehner/only-brain/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/only-brain.svg)](LICENSE)

A small feed-forward neural network library for Rust, **without the learning part**.

Only Brain gives you the network: fully connected layers, an activation function
per layer, a forward pass, and direct access to every weight and bias. How those
weights are found is up to you. Backpropagation, a genetic algorithm, random search
or a formula on a napkin all work, because the library never hides the numbers.

- **Shape in the type.** `NeuralNetwork<IN, OUT>` makes a wrongly sized input a
  compile error. Hidden layers are chosen at runtime.
- **Plain data at the boundary.** Weights and biases go in and out as arrays, slices
  and `Vec`s. You never touch a matrix type.
- **Two views of the same network.** Read and write it layer by layer, or as one
  flat list of parameters for black-box search.
- **Stable on disk.** A small, versioned binary format with no serialization
  dependency, plus serde for everything else. Old files keep loading.

## Installation

```sh
cargo add only-brain
```

## Quick start

```rust
use only_brain::NeuralNetwork;

fn main() {
    // A 2 -> 2 -> 1 network. Layer 0 is the input layer.
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

`new` draws the weights uniformly from `[-1, 1)` with zero biases. Use
`new_with_rng` for a seeded, reproducible start.

## Training it your way

### Layer by layer

For methods that reason about layers, such as backpropagation, every layer's
weights and biases can be read and replaced, and single values can be touched
directly. Layer 0 is the input layer, so the first layer with weights is layer 1.

```rust
use only_brain::NeuralNetwork;

let mut nn = NeuralNetwork::<2, 1>::new(&[2]);

// Read a layer's weights, adjust them, write them back.
let mut rows = nn.layer_weights(1);
rows[0][1] += 0.01;
nn.set_layer_weights(1, &rows);

// Or address one weight or bias: (layer, neuron, input) and (layer, neuron).
nn.set_weight(2, 0, 1, 0.5);
nn.set_bias(2, 0, -0.1);
```

### As a flat list of parameters

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

`set_parameters` updates an existing network in place and keeps its activation
functions, which are part of the shape rather than the parameters.
`NeuralNetwork` is `Send + Sync`, so a population can be scored in parallel. See
[`examples/xor_evolution.rs`](https://github.com/IgorFroehner/only-brain/blob/main/examples/xor_evolution.rs)
for a complete, seeded neuroevolution run.

## Activation functions

Every layer has its own activation function, so hidden layers and the output layer
can differ. The available ones are `Sigmoid` (the default), `Tanh`, `ReLU`,
`BinaryStep` and `Identity`.

```rust
use only_brain::{ActivationFunction, NeuralNetwork};

let mut nn = NeuralNetwork::<10, 3>::new(&[8]);
nn.set_activation_function(ActivationFunction::ReLU); // every layer
nn.set_output_activation(ActivationFunction::Tanh);   // then just the output
nn.set_layer_activation(1, ActivationFunction::Sigmoid); // or one layer
```

## Saving and loading

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

Models are stored in a small, versioned binary format, documented in
[`src/io.rs`](https://github.com/IgorFroehner/only-brain/blob/main/src/io.rs), that
does not depend on any serialization library, so saved models keep loading as the
library evolves. Files written by 0.1 and 0.2 still load. `write_model` and
`read_model` do the same with any `Write` or `Read`, such as a `Vec<u8>`.

Every failure is a `ModelError` that says what went wrong: a file that is not a
model, one written by a newer version, a truncated file, or a shape that does not
match the requested type.

### serde

`NeuralNetwork` also implements serde's `Serialize` and `Deserialize` through a
plain form, `{"layers": [{"activation", "weights", "biases"}]}`, so it can be
embedded in your own types and formats. Deserializing validates the shape too.

```rust
use only_brain::NeuralNetwork;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let nn = NeuralNetwork::<2, 1>::from_parameters(&[], &[0.5, -0.25, 0.1]);

let json = serde_json::to_string(&nn)?;
assert_eq!(json, r#"{"layers":[{"activation":"Sigmoid","weights":[[0.5,-0.25]],"biases":[0.1]}]}"#);

let back: NeuralNetwork<2, 1> = serde_json::from_str(&json)?;
let back: NeuralNetwork<2, 1> = serde_json::from_str(&json)?;

assert_eq!(back, nn);
# Ok(())
# }
```

## Perceptron

A single neuron over a compile-time-sized vector, for the classics.

```rust
use only_brain::{bvector, ActivationFunction, Perceptron};

let mut and_gate = Perceptron::<2>::new(ActivationFunction::BinaryStep);
and_gate.set_weights(bvector![1.0, 1.0]);
and_gate.set_bias(-1.5);

assert_eq!(and_gate.feed_forward(&bvector![1.0, 1.0]), 1.0);
assert_eq!(and_gate.feed_forward(&bvector![1.0, 0.0]), 0.0);
```

## Examples

| Example | Shows |
|---|---|
| `neural_network` | Building a network by hand and running it. |
| `dump_load` | Saving and loading a model. |
| `xor_evolution` | Evolving a network with a genetic algorithm through the flat parameter view. |
| `perceptron` | A single perceptron. |
| `perceptron_iris` | Training a perceptron on the Iris dataset with the perceptron rule. |

Run one with `cargo run --example xor_evolution`.

## Development

```sh
cargo test --all-targets                      # unit and integration tests, builds the examples
cargo test --doc                              # doctests, including every snippet in this README
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo bench                                   # feed_forward, flat parameters, model IO
```

CI runs the same checks on every push and pull request. The README is compiled as
doctests, so its examples cannot go stale.

## Changelog

See [CHANGELOG.md](https://github.com/IgorFroehner/only-brain/blob/main/CHANGELOG.md).

## License

MIT. See [LICENSE](https://github.com/IgorFroehner/only-brain/blob/main/LICENSE).

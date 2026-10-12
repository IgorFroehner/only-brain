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

That range saturates sigmoid and tanh in deeper networks, so `new_with_init` takes an
`Init` that scales each layer's range to its shape: `Init::Xavier` for sigmoid and
tanh layers, `Init::He` for ReLU layers, or `Init::Uniform { low, high }` for a fixed
range.

```rust
use only_brain::{ActivationFunction, Init, NeuralNetwork};
use rand::{rngs::StdRng, SeedableRng};

let mut rng = StdRng::seed_from_u64(42);
let mut nn = NeuralNetwork::<16, 4>::new_with_init(&[64, 64], Init::He, &mut rng);
nn.set_activation_function(ActivationFunction::ReLU);
nn.set_output_activation(ActivationFunction::Sigmoid);
```

To run many inputs through the same network, such as a whole dataset,
`feed_forward_batch` takes a slice of inputs and returns one output per input. It
computes the same values as `feed_forward` but runs each layer as one matrix product
over every input, which is faster from a few dozen inputs up.

```rust
use only_brain::NeuralNetwork;

let nn = NeuralNetwork::<2, 1>::new(&[2]);

let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
let outputs = nn.feed_forward_batch(&inputs);

assert_eq!(outputs.len(), 4);
```

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
assert_eq!(nn.weight(2, 0, 1), 0.5);
assert_eq!(nn.bias(2, 0), -0.1);
```

Backpropagation also needs what the forward pass computed at every layer.
`feed_forward_trace` runs the network once and keeps each layer's pre-activations
(`W·x + b`) and outputs, and `ActivationFunction::derivative` gives the slope of each
activation at a pre-activation. With the layer getters and setters, that is a full
gradient descent step:

```rust
use only_brain::{ActivationFunction, NeuralNetwork};

// A 2 -> 2 -> 1 network with known starting weights.
let mut nn = NeuralNetwork::<2, 1>::from_parameters(&[2], &[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.5, -0.5, 0.0]);
nn.set_activation_function(ActivationFunction::Tanh);

let (input, target, learning_rate) = ([1.0, 0.5], 0.8, 0.1);
let loss = |nn: &NeuralNetwork<2, 1>| (nn.feed_forward(&input)[0] - target).powi(2) / 2.0;
let before = loss(&nn);

// Forward, keeping every layer's pre-activations and outputs.
let trace = nn.feed_forward_trace(&input);

// Backward, from the output layer to layer 1. `gradient` is the loss's gradient
// with respect to the current layer's outputs.
let mut gradient = vec![trace.output()[0] - target];
for layer in (1..nn.num_layers()).rev() {
    let activation = nn.layer_activation(layer);
    let delta: Vec<f64> = gradient.iter().zip(trace.pre_activations(layer))
        .map(|(g, &z)| g * activation.derivative(z))
        .collect();

    let inputs = trace.outputs(layer - 1); // layer 0's outputs are the inputs
    let mut weights = nn.layer_weights(layer);
    if layer > 1 {
        // Pass the gradient back to the previous layer (layer 1's is the inputs).
        gradient = (0..inputs.len())
            .map(|i| delta.iter().zip(&weights).map(|(d, row)| d * row[i]).sum())
            .collect();
    }

    for (row, d) in weights.iter_mut().zip(&delta) {
        for (w, x) in row.iter_mut().zip(inputs) {
            *w -= learning_rate * d * x;
        }
    }
    let biases: Vec<f64> = nn.layer_biases(layer).iter().zip(&delta)
        .map(|(b, d)| b - learning_rate * d)
        .collect();
    nn.set_layer_weights(layer, &weights);
    nn.set_layer_biases(layer, &biases);
}

assert!(loss(&nn) < before);
```

See
[`examples/iris_backprop.rs`](https://github.com/IgorFroehner/only-brain/blob/main/examples/iris_backprop.rs)
for a classifier trained this way.

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
`LeakyReLU` (slope 0.01 below zero), `ELU`, `Softplus`, `SiLU` (swish), `BinaryStep`
and `Identity`.

```rust
use only_brain::{ActivationFunction, NeuralNetwork};

let mut nn = NeuralNetwork::<10, 3>::new(&[8]);
nn.set_activation_function(ActivationFunction::ReLU); // every layer
nn.set_output_activation(ActivationFunction::Tanh);   // then just the output
nn.set_layer_activation(1, ActivationFunction::Sigmoid); // or one layer

// Evaluate one on its own.
assert_eq!(ActivationFunction::ReLU.apply(-2.0), 0.0);
```

Softmax is not an activation function, because each of its outputs depends on the
whole layer. For a classifier, use an `Identity` output layer and apply `softmax` to
the outputs, then `argmax` to pick the class:

```rust
use only_brain::{argmax, softmax, ActivationFunction, NeuralNetwork};

let mut classifier = NeuralNetwork::<4, 3>::new(&[8]);
classifier.set_activation_function(ActivationFunction::ReLU);
classifier.set_output_activation(ActivationFunction::Identity);

let probabilities = softmax(&classifier.feed_forward(&[5.1, 3.5, 1.4, 0.2]));
let class = argmax(&probabilities);
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

assert_eq!(back, nn);
# Ok(())
# }
```

## Perceptron

A single neuron with `N` inputs, for the classics. Like the network, it takes plain
arrays, so the input width is checked at compile time.

```rust
use only_brain::{ActivationFunction, NeuralNetwork, Perceptron};

let mut and_gate = Perceptron::<2>::new(ActivationFunction::BinaryStep);
and_gate.set_weights([1.0, 1.0]);
and_gate.set_bias(-1.5);

assert_eq!(and_gate.feed_forward(&[1.0, 1.0]), 1.0);
assert_eq!(and_gate.feed_forward(&[1.0, 0.0]), 0.0);
assert_eq!(and_gate.weights(), [1.0, 1.0]);

// A perceptron is a network with no hidden layers, which is how it is saved.
let network = NeuralNetwork::from(and_gate);
assert_eq!(network.feed_forward(&[1.0, 1.0]), [1.0]);
```

`new_with_rng` starts it from random weights, and it implements serde through the
plain form `{"activation", "weights", "bias"}`. To save one with `dump_model`,
convert it to a `NeuralNetwork<N, 1>` as above; it loads back as that network.

## Examples

| Example | Shows |
|---|---|
| `neural_network` | Building a network by hand and running it. |
| `dump_load` | Saving and loading a model. |
| `xor_evolution` | Evolving a network with a genetic algorithm through the flat parameter view. |
| `iris_backprop` | Training a classifier on the Iris dataset with backpropagation, through the traced forward pass. |
| `perceptron` | A single perceptron. |
| `perceptron_iris` | Training a perceptron on the Iris dataset with the perceptron rule. |

Run one with `cargo run --example xor_evolution`.

## Development

```sh
cargo test --all-targets                      # unit and integration tests, builds the examples
cargo test --doc                              # doctests, including every snippet in this README
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo bench                                   # feed_forward, batches, flat parameters, model IO
```

CI runs the same checks on every push and pull request. The README is compiled as
doctests, so its examples cannot go stale.

## Changelog

See [CHANGELOG.md](https://github.com/IgorFroehner/only-brain/blob/main/CHANGELOG.md).

## License

MIT. See [LICENSE](https://github.com/IgorFroehner/only-brain/blob/main/LICENSE).

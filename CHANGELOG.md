# Changelog

## Unreleased

### Added

- `feed_forward_batch`, which runs many inputs through one network as a single
  matrix product per layer: 28% faster than a loop of `feed_forward` calls on a
  16-16-16-4 network and 2.1x faster on a 64-128-64-10 network, for 256 inputs.
  A benchmark group compares the two.

### Changed

- `feed_forward` keeps its scratch space on the stack for networks whose widest
  layer has at most 16 neurons: 12% faster on a 4-6-2 network, unchanged above
  that width.

## 0.3.1

Faster `feed_forward` and model files, with no interface changes.

### Added

- `BVector::is_empty`.
- Criterion benchmarks (`cargo bench`) for `feed_forward`, the flat parameter view
  and model reading and writing.

### Changed

- `feed_forward` allocates once per call instead of twice per layer: 22% to 54%
  faster depending on the network's shape.
- `write_model` writes straight from the layers instead of copying every weight
  first: 42% faster on a 64-128-64-10 network.
- `read_model` reads up to 512 floats at a time instead of one: 23% faster on a
  64-128-64-10 network.

### Fixed

- The `bvector!` examples named a `barray!` macro that doesn't exist.

## 0.3.0

Ready to be evolved: everything a genetic algorithm needs to search a network's
weights.

0.2.0 was never published to crates.io, so this is the first release after 0.1.4
and it also carries the 0.2.0 changes listed below. Upgrading from 0.1.4 means
reading both sections.

### Added

- Flat parameter view: `parameter_count`, `parameter_count_for`, `parameters`,
  `set_parameters` and `from_parameters`, in a documented, stable order (layer by
  layer, weights row by row, then biases).
- Activation function per layer: `layer_activation`, `set_layer_activation`,
  `set_output_activation` and `output_activation`. `set_activation_function` now
  sets every layer.
- `ActivationFunction::Identity`, `ActivationFunction::apply` and
  `ActivationFunction::ALL`.
- `hidden_layer_sizes`, to rebuild a network from its parameters.
- `write_model` and `read_model`, to save and load with any writer or reader.
- A versioned model format (`MODEL_FORMAT_VERSION`), with new
  `ModelError::UnrecognizedFormat`, `UnsupportedVersion` and `Malformed` errors.
- `examples/xor_evolution.rs`, a seeded neuroevolution run.

### Changed

- **Breaking:** `activation_function()` is gone, since each layer now has its own;
  use `layer_activation` or `output_activation`.
- **Breaking:** `ActivationFunction` is `#[non_exhaustive]`.
- **Breaking:** models are saved in the new versioned format, which 0.2 cannot read.
  Files written by 0.1 and 0.2 still load, with their network-wide activation applied
  to every layer.
- **Breaking:** the serde form of `NeuralNetwork` is now a plain list of layers with
  `activation`, `weights` (one row per neuron) and `biases`, independent of nalgebra.
- **Breaking:** `ModelError::Serialization` is replaced by the errors above.
- `bincode` is no longer a dependency, and nalgebra no longer needs its serde
  feature.
- `Perceptron` derives `Debug`, `Clone` and `PartialEq`.

### Removed

- The empty `bmatrix` module.
- The polars dev-dependency: `examples/polars.rs` is now
  `examples/perceptron_iris.rs`, which reads the CSV with the standard library.

## 0.2.0

Never published; its changes ship in 0.3.0.

### Added

- `Perceptron`, a single neuron over a compile-time-sized `BVector`, with the
  `bvector!` macro.
- `set_activation_function`, which had no setter before, so every network ran
  sigmoid.
- `new_with_rng`, for reproducible initialisation.
- `layer_weights`, `layer_biases`, `set_bias` and `get_bias`.
- `ModelError`, a typed error for saving and loading.
- `Debug`, `Clone` and `PartialEq` on the public types.

### Changed

- **Breaking:** the input and output widths are part of the type:
  `NeuralNetwork<IN, OUT>`. `new` takes only the hidden layer sizes as a slice, and
  `feed_forward` takes `&[f64; IN]` and returns `[f64; OUT]`, so a wrongly sized
  input is a compile error.
- **Breaking:** `set_layer_weights` takes one row per neuron as nested arrays or
  `Vec<Vec<f64>>`, and `set_layer_biases` takes `&[f64]`, instead of nalgebra's
  `DMatrix` and `DVector`.
- **Breaking:** `dump_model` and `load_model` take `impl AsRef<Path>` and return
  `ModelError` instead of `Box<dyn Error>`.
- Loading checks that every layer fits its neighbours and the requested widths,
  instead of panicking later in `feed_forward`. Deserializing through serde runs the
  same checks.
- Reads and writes are buffered.
- rand 0.9 and nalgebra 0.34.

### Fixed

- A network using `BinaryStep` panicked on every forward pass.
- `Display` printed a function pointer address instead of the activation's name.

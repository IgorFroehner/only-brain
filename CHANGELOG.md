# Changelog

## 0.3.0

Ready to be evolved: everything a genetic algorithm needs to search a network's
weights.

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

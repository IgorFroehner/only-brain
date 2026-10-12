# Changelog

## 0.5.0

Plain arrays at the whole boundary and a `Perceptron` on par with `NeuralNetwork`;
what backpropagation needs (a traced forward pass and activation derivatives); four
new activation functions and a `softmax` helper; scaled initialisation and tidier
names; in-place parameter mutation, and weights stored in the flat parameter order.

### Added

- `Init` and `NeuralNetwork::new_with_init(hidden, init, rng)`: weights drawn from
  `Init::Xavier` (Glorot uniform, for sigmoid and tanh), `Init::He` (He uniform, for
  ReLU) or `Init::Uniform { low, high }`. `Init::default()` is the `[-1, 1)` of `new`,
  and `new` and `new_with_rng` produce the same weights for a seed as before.
- `NeuralNetwork::weight(layer, neuron, input)` and `NeuralNetwork::bias(layer,
  neuron)`.
- `Perceptron::weights`, `bias`, `activation_function`, `set_activation_function`,
  `new_with_rng` (the weights `NeuralNetwork::<N, 1>::new_with_rng(&[], rng)` would
  draw), `Default` (zero weights and bias, sigmoid), and serde `Serialize` /
  `Deserialize` through the plain form `{"activation", "weights", "bias"}`, which
  checks the number of weights.
- `From<Perceptron<N>> for NeuralNetwork<N, 1>`, a network with no hidden layers and
  the same weights, bias and activation. Save a perceptron by converting it and calling
  `dump_model`; it loads back as that network. Its outputs agree with the perceptron's
  up to floating-point rounding, so a `BinaryStep` perceptron can classify inputs
  within rounding of its boundary differently.
- `From<BVector<T, N>> for [T; N]`, to move existing code off `BVector`.
- `NeuralNetwork::feed_forward_trace`, a forward pass that returns a `Trace` of every
  layer's pre-activations (`W·x + b`) and outputs, numbered like the rest of the API
  (layer 0 is the input layer, whose outputs are the inputs). Its output is exactly
  what `feed_forward` returns.
- `ActivationFunction::derivative`, the derivative of each activation with respect to
  its pre-activation. `ReLU`'s derivative is 0 at 0, and `BinaryStep`'s is 0
  everywhere, so it cannot be trained by gradient descent.
- The `iris_backprop` example, which trains a 4-8-3 classifier on the Iris dataset
  with backpropagation and SGD written against the public API, reaching 96.7% test
  accuracy. CI runs it.
- `ActivationFunction::LeakyReLU` (slope fixed at 0.01 for `x < 0`),
  `ActivationFunction::ELU` (`α = 1`), `ActivationFunction::Softplus` (`ln(1 + e^x)`,
  computed so it neither overflows nor underflows early for large `|x|`) and
  `ActivationFunction::SiLU` (swish, `x · sigmoid(x)`), with their derivatives. They
  are stored in model files as activation codes 5, 6, 7 and 8, and the format version
  stays 1. Compatibility: a 0.3 or 0.4 reader reports a model that uses one of them
  as `ModelError::Malformed` ("unknown activation function code"), and JSON or other
  serde output naming one of them does not deserialize in 0.4. Models using only the
  older functions are byte-for-byte unchanged and still load everywhere.
- `softmax(&[f64; N]) -> [f64; N]`, a numerically stable softmax (it subtracts the
  maximum first), and `argmax(&[f64; N]) -> usize`, the index of the largest value.
  Softmax is not an `ActivationFunction`, because each output depends on the whole
  layer; the pattern is an `Identity` output layer and
  `softmax(&nn.feed_forward(&x))`. The `iris_backprop` example uses both.
- `NeuralNetwork::for_each_parameter_mut(f)` and `NeuralNetwork::map_parameters(f)`,
  which visit every weight and bias in place, in the documented flat parameter
  order, and keep the activation functions. A genetic algorithm that keeps networks
  rather than genomes can mutate a child without `parameters()` and
  `set_parameters()` copying everything out and back: negating every parameter of a
  16-16-16-4 network takes 41 ns in place against 152 ns through the round trip.
  The `parameters` benchmark group compares the two.

### Changed

- **Breaking:** `Perceptron::set_weights` takes `[f64; N]` and
  `Perceptron::feed_forward` takes `&[f64; N]` instead of `BVector`s. Migration:
  replace `bvector![a, b]` with `[a, b]`, or convert an existing `BVector` with
  `.into()`.
- **Breaking:** `BVector::data_vector` is private, so no nalgebra type appears in a
  public signature or field, apart from the `T: SimdRealField` bound on the
  deprecated `BVector`. Migration: read the elements with `get(i)` or convert with
  `let array: [f64; N] = vector.into();`.
- **Breaking:** the free functions `sigmoid`, `tanh`, `relu`, `binary_step`,
  `identity` and `get_activation_function` are no longer exported. Migration: call
  `ActivationFunction::Sigmoid.apply(x)` (and so on) instead.
- `Perceptron`'s `Display` prints its weights as a plain list.
- Deprecated `BVector` and `bvector!`: plain arrays replace them, and they will be
  removed in a later release.
- Deprecated `NeuralNetwork::get_weight` and `NeuralNetwork::get_bias` in favour of
  `weight` and `bias`. They still work. Migration: drop the `get_` prefix.
- `feed_forward` and `feed_forward_batch` add the biases and apply the activation
  function in one loop specialised for each function, instead of calling it through
  a function pointer once per neuron. With sigmoid layers, `feed_forward` is 4%
  faster on a 4-6-2 network, 5% on a 16-16-16-4 network and 3% on a 64-128-64-10
  network, and `feed_forward_batch` is 18% faster on a 16-16-16-4 network and 13%
  on a 64-128-64-10 network, for 256 inputs. A new benchmark group,
  `feed_forward_by_activation`, runs the 16-16-16-4 network with each function:
  every one is faster, from 3% for tanh and 4% for softplus to 11% for SiLU.
- Each layer stores its weights transposed (one column per neuron), so in memory
  they are already in the flat parameter order. `parameters`, `set_parameters`,
  `from_parameters` and `write_model` copy whole slices instead of one weight at a
  time, and `feed_forward` computes one dot product of contiguous weights per
  neuron. On a 16-16-16-4 network `parameters` is 87% faster, `set_parameters` 76%
  and `from_parameters` 44%; on a 64-128-64-10 network `write_model` is 52% faster
  and `read_model` 34%. `feed_forward` is 26% faster on a 4-6-2 network, 21% on a
  16-16-16-4 network and 17% on a 64-128-64-10 network. `feed_forward_batch` is
  unchanged within noise (-1% on 16-16-16-4, +3% on 64-128-64-10, for 256 inputs):
  it multiplies by a transposed copy of each layer's weights, because nalgebra's
  transposed product does not use its fast matrix multiplication and was 85% slower
  on the larger network. Numbers are the best of four alternating runs against the
  previous layout. The flat parameter order, the model file format and the weights
  `new` and `new_with_rng` draw for a seed are unchanged.

### Fixed

- A pre-0.3 file whose activation tag is 4 or higher is reported as
  `ModelError::UnrecognizedFormat`. Only tags 0 to 3 (sigmoid, tanh, ReLU, binary step)
  existed in that format, but the reader looked tags up in the current code table, so
  such a file loaded with `Identity` or one of the new functions.

### Removed

- **Breaking:** `NeuralNetwork::print`. Migration: `println!("{nn}")`, which
  `Display` already formats.

## 0.4.0

Batch inference, and a faster `feed_forward` for small networks.

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

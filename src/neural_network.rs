use crate::activation_functions::ActivationFunction;
use crate::layer::Layer;
use crate::ModelError;
use nalgebra::DMatrix;
use rand::{rng, Rng};
use std::fmt;
use serde::{Deserialize, Serialize};

/// Neural Network
///
/// This is the main struct of the library: a chain of fully connected layers, each with
/// its own weights, biases and [`ActivationFunction`]. You can use this struct and its
/// methods to create, manipulate and even implement your own ways to train a neural
/// network.
///
/// The number of input neurons (`IN`) and output neurons (`OUT`) are part of the type,
/// so feeding a wrongly sized input is a compile error rather than a runtime panic. The
/// hidden layers stay dynamic and are given at construction time.
///
/// # Layers
///
/// Layer 0 is the input layer, which only passes the inputs on, so it has no weights,
/// biases or activation. Layers `1..num_layers()` each hold one row of weights per
/// neuron, with one weight per neuron of the previous layer, and one bias per neuron.
///
/// # Example
///
/// ```
/// use only_brain::NeuralNetwork;
///
/// // A 2 -> 2 -> 1 network.
/// let mut nn = NeuralNetwork::<2, 1>::new(&[2]);
///
/// // One row per neuron, one weight per neuron of the previous layer.
/// nn.set_layer_weights(1, &[[0.1, 0.2],
///                           [0.3, 0.4]]);
/// nn.set_layer_biases(1, &[0.1, 0.2]);
///
/// nn.set_layer_weights(2, &[[0.9, 0.8]]);
/// nn.set_layer_biases(2, &[0.1]);
///
/// let output = nn.feed_forward(&[0.5, 0.2]);
///
/// println!("{:?}", output);
/// ```
///
/// # Flat parameter view
///
/// Search methods such as genetic algorithms usually work on a flat list of numbers
/// rather than on layers. [`parameters`](Self::parameters),
/// [`set_parameters`](Self::set_parameters) and
/// [`from_parameters`](Self::from_parameters) convert between a network and such a
/// list, in an order that is documented and stable across versions:
///
/// - layer by layer, from layer 1 to the output layer;
/// - within a layer, every weight first, row by row (all the weights of neuron 0, then
///   of neuron 1, and so on, each row in the order of the previous layer's neurons);
/// - then that layer's biases, one per neuron.
///
/// Activation functions are part of the network's shape, not of its parameters, so
/// they are left untouched by [`set_parameters`](Self::set_parameters).
///
/// ```
/// use only_brain::NeuralNetwork;
///
/// let mut nn = NeuralNetwork::<2, 1>::new(&[]);
/// nn.set_layer_weights(1, &[[0.5, -0.25]]);
/// nn.set_layer_biases(1, &[0.1]);
///
/// assert_eq!(nn.parameters(), vec![0.5, -0.25, 0.1]);
///
/// let rebuilt = NeuralNetwork::<2, 1>::from_parameters(&[], &nn.parameters());
/// assert_eq!(rebuilt, nn);
/// ```
///
/// # Serialization
///
/// Networks implement serde's `Serialize` and `Deserialize` through a plain form, a
/// list of layers each with its `activation`, `weights` (one row per neuron) and
/// `biases`, so they can be embedded in your own types and formats. Deserializing checks
/// that the layers fit together and match `IN` and `OUT`. To save a network to a file,
/// see [`crate::dump_model`] and [`crate::load_model`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "NetworkData", try_from = "NetworkData")]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
}

/// The plain form of a network, before its shape has been checked.
///
/// Serde and the model file readers both go through this, so every entry point, not
/// only [`crate::load_model`], rejects layers that do not fit together. It is also what
/// keeps the serialized form independent of how layers are stored in memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct NetworkData {
    pub(crate) layers: Vec<LayerData>,
}

/// The plain form of one layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LayerData {
    pub(crate) activation: ActivationFunction,
    /// One row per neuron, one weight per neuron of the previous layer.
    pub(crate) weights: Vec<Vec<f64>>,
    pub(crate) biases: Vec<f64>,
}

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// Creates a new Neural Network with the given hidden layer sizes. The input and
    /// output widths come from the type parameters, so `hidden` lists only the layers
    /// between them and may be empty.
    ///
    /// Weights are initialised uniformly at random in `[-1, 1)`, biases at zero, and
    /// every layer uses [`ActivationFunction::Sigmoid`]. Use
    /// [`NeuralNetwork::new_with_rng`] to control the seed.
    ///
    /// # Panics
    ///
    /// Panics if any hidden layer size is zero. `IN` or `OUT` being zero is a compile
    /// error.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// // 2 -> 2 -> 1
    /// let nn = NeuralNetwork::<2, 1>::new(&[2]);
    ///
    /// // 3 -> 1, no hidden layers
    /// let direct = NeuralNetwork::<3, 1>::new(&[]);
    /// ```
    pub fn new(hidden: &[usize]) -> Self {
        Self::new_with_rng(hidden, &mut rng())
    }

    /// Creates a new Neural Network using the given random number generator.
    ///
    /// Seeding the generator makes initialisation reproducible, which is what you want
    /// in tests and when a training run needs to be repeatable.
    ///
    /// # Panics
    ///
    /// Panics if any hidden layer size is zero.
    pub fn new_with_rng<R: Rng>(hidden: &[usize], rng: &mut R) -> Self {
        Self::build(hidden, |neurons, inputs| {
            Layer::random(neurons, inputs, ActivationFunction::default(), rng)
        })
    }

    /// Creates a network with the given hidden layer sizes from a flat list of
    /// parameters, in the order described in the [flat parameter
    /// view](Self#flat-parameter-view). Every layer uses
    /// [`ActivationFunction::Sigmoid`]; change that afterwards with
    /// [`set_activation_function`](Self::set_activation_function) and friends.
    ///
    /// # Panics
    ///
    /// Panics if any hidden layer size is zero, or if `parameters` does not hold exactly
    /// [`parameter_count_for(hidden)`](Self::parameter_count_for) values.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, NeuralNetwork};
    /// // A genome from a genetic algorithm, for a 2 -> 2 -> 1 network.
    /// let genome = vec![0.5; NeuralNetwork::<2, 1>::parameter_count_for(&[2])];
    ///
    /// let mut nn = NeuralNetwork::<2, 1>::from_parameters(&[2], &genome);
    /// nn.set_output_activation(ActivationFunction::Tanh);
    ///
    /// let [steering] = nn.feed_forward(&[0.3, -0.8]);
    /// assert!((-1.0..=1.0).contains(&steering));
    /// ```
    pub fn from_parameters(hidden: &[usize], parameters: &[f64]) -> Self {
        let mut network = Self::build(hidden, |neurons, inputs| {
            Layer::zeros(neurons, inputs, ActivationFunction::default())
        });
        network.set_parameters(parameters);
        network
    }

    /// Chains `IN`, the hidden sizes and `OUT` into layers made by `make(neurons, inputs)`.
    fn build(hidden: &[usize], mut make: impl FnMut(usize, usize) -> Layer) -> Self {
        let layers = Self::layer_shapes(hidden)
            .map(|(neurons, inputs)| make(neurons, inputs))
            .collect();

        Self { layers }
    }

    /// The `(neurons, inputs)` of every layer after the input layer.
    fn layer_shapes(hidden: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
        const { assert!(IN > 0, "a network needs at least one input neuron") };
        const { assert!(OUT > 0, "a network needs at least one output neuron") };
        assert!(
            hidden.iter().all(|&size| size > 0),
            "every hidden layer must have at least one neuron, got {hidden:?}"
        );

        let inputs = std::iter::once(IN).chain(hidden.iter().copied());
        let neurons = hidden.iter().copied().chain(std::iter::once(OUT));
        neurons.zip(inputs)
    }

    /// Feeds the given inputs to the neural network and returns the output.
    ///
    /// The input and output widths are checked at compile time.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let mut nn = NeuralNetwork::<1, 1>::new(&[]);
    ///
    /// nn.set_layer_weights(1, &[[0.5]]);
    /// nn.set_layer_biases(1, &[0.5]);
    ///
    /// let output = nn.feed_forward(&[0.5]);
    /// assert!((output[0] - 0.679178699175393).abs() < 1e-12);
    /// ```
    pub fn feed_forward(&self, inputs: &[f64; IN]) -> [f64; OUT] {
        // Two halves that the layers ping-pong between, each wide enough for the
        // widest layer. Networks up to STACK_WIDTH neurons per layer, which is every
        // network a genome is likely to drive, run without touching the heap.
        const STACK_WIDTH: usize = 16;

        let widest = self.widest_layer();
        if widest <= STACK_WIDTH {
            let mut scratch = [0.0; STACK_WIDTH * 2];
            self.forward_in(inputs, &mut scratch[..widest * 2])
        } else {
            let mut scratch = vec![0.0; widest * 2];
            self.forward_in(inputs, &mut scratch)
        }
    }

    /// The width of the widest layer, counting the input layer.
    fn widest_layer(&self) -> usize {
        self.layers.iter().map(Layer::size).fold(IN, usize::max)
    }

    /// Runs the layers over `scratch`, which must hold `2 * widest_layer()` values.
    #[inline(always)]
    fn forward_in(&self, inputs: &[f64; IN], scratch: &mut [f64]) -> [f64; OUT] {
        let (mut current, mut next) = scratch.split_at_mut(scratch.len() / 2);

        current[..IN].copy_from_slice(inputs);
        let mut width = IN;

        for layer in &self.layers {
            layer.forward_into(&current[..width], &mut next[..layer.size()]);
            width = layer.size();
            std::mem::swap(&mut current, &mut next);
        }

        // The layer chain is built from IN..OUT and every setter checks its dimensions,
        // so the final slice always has exactly OUT elements.
        <[f64; OUT]>::try_from(&current[..width])
            .expect("output layer width should match the OUT type parameter")
    }

    /// Feeds every input to the network and returns one output per input, in order.
    ///
    /// This computes the same values as calling [`feed_forward`](Self::feed_forward)
    /// once per input, but runs each layer as a single matrix product over all the
    /// inputs at once, which is faster when there are many of them: scoring a dataset,
    /// for example. It does not help when every input goes to a different network.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let nn = NeuralNetwork::<2, 1>::new(&[3]);
    ///
    /// let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    /// let outputs = nn.feed_forward_batch(&inputs);
    ///
    /// assert_eq!(outputs.len(), 4);
    /// assert_eq!(outputs[2], nn.feed_forward(&inputs[2]));
    /// ```
    pub fn feed_forward_batch(&self, inputs: &[[f64; IN]]) -> Vec<[f64; OUT]> {
        let samples = inputs.len();
        let widest = self.widest_layer();

        // One column per sample, with as many rows as the widest layer; each layer
        // works on the top rows it needs and the two matrices ping-pong.
        let mut current = DMatrix::zeros(widest, samples);
        let mut next = DMatrix::zeros(widest, samples);

        for (mut column, sample) in current.column_iter_mut().zip(inputs) {
            column.rows_mut(0, IN).copy_from_slice(sample);
        }
        let mut width = IN;

        for layer in &self.layers {
            layer.forward_batch_into(current.rows(0, width), next.rows_mut(0, layer.size()));
            width = layer.size();
            std::mem::swap(&mut current, &mut next);
        }

        current
            .rows(0, width)
            .column_iter()
            // As in `feed_forward`, the last layer is always OUT wide.
            .map(|column| std::array::from_fn(|neuron| column[neuron]))
            .collect()
    }

    /// Returns the number of weights and biases in the network, which is the length of
    /// [`parameters`](Self::parameters).
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(Layer::parameter_count).sum()
    }

    /// Returns the number of weights and biases a network with these hidden layer sizes
    /// has, without building one. This is the genome length to give a genetic algorithm.
    ///
    /// # Panics
    ///
    /// Panics if any hidden layer size is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// // 10 -> 8 -> 3: (10 + 1) * 8 + (8 + 1) * 3
    /// assert_eq!(NeuralNetwork::<10, 3>::parameter_count_for(&[8]), 115);
    /// ```
    pub fn parameter_count_for(hidden: &[usize]) -> usize {
        Self::layer_shapes(hidden)
            .map(|(neurons, inputs)| Layer::parameter_count_for(neurons, inputs))
            .sum()
    }

    /// Returns every weight and bias as one flat list, in the order described in the
    /// [flat parameter view](Self#flat-parameter-view).
    pub fn parameters(&self) -> Vec<f64> {
        let mut parameters = Vec::with_capacity(self.parameter_count());
        for layer in &self.layers {
            layer.extend_parameters(&mut parameters);
        }
        parameters
    }

    /// Replaces every weight and bias from one flat list, in the order described in the
    /// [flat parameter view](Self#flat-parameter-view). Activation functions are kept.
    ///
    /// # Panics
    ///
    /// Panics if `parameters` does not hold exactly
    /// [`parameter_count()`](Self::parameter_count) values.
    pub fn set_parameters(&mut self, parameters: &[f64]) {
        let expected = self.parameter_count();
        assert_eq!(
            parameters.len(),
            expected,
            "Incompatible parameter count: expected {expected} parameters, got {}",
            parameters.len()
        );

        let mut rest = parameters;
        for layer in &mut self.layers {
            rest = layer.take_parameters(rest);
        }
    }

    /// Sets every weight of the given layer, as one row per neuron of that layer
    /// holding one weight per neuron of the previous layer.
    ///
    /// Layer 0 is the input layer, which has no weights, so `layer` starts at 1. Nested
    /// arrays and `Vec<Vec<f64>>` are both accepted.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0 or past the output layer, or if `weights` does not have
    /// exactly `layer_size(layer)` rows of `layer_size(layer - 1)` weights.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let mut nn = NeuralNetwork::<3, 1>::new(&[2]);
    ///
    /// nn.set_layer_weights(1, &[[0.1, 0.2, 0.3],
    ///                           [0.4, 0.5, 0.6]]);
    ///
    /// // Weights computed at runtime work the same way.
    /// let output_weights = vec![vec![0.7, 0.8]];
    /// nn.set_layer_weights(2, &output_weights);
    ///
    /// assert_eq!(nn.layer_weights(2), output_weights);
    /// ```
    pub fn set_layer_weights<R: AsRef<[f64]>>(&mut self, layer: usize, weights: &[R]) {
        self.layer_mut(layer).set_weights(weights);
    }

    /// Returns the weights of the given layer, as one row per neuron of that layer
    /// holding one weight per neuron of the previous layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0 or past the output layer.
    pub fn layer_weights(&self, layer: usize) -> Vec<Vec<f64>> {
        self.layer(layer).weight_rows()
    }

    /// Sets the biases of the given layer, one per neuron.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0 or past the output layer, or if `biases` does not have
    /// exactly `layer_size(layer)` elements.
    pub fn set_layer_biases(&mut self, layer: usize, biases: &[f64]) {
        self.layer_mut(layer).set_biases(biases);
    }

    /// Returns the biases of the given layer, one per neuron.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0 or past the output layer.
    pub fn layer_biases(&self, layer: usize) -> &[f64] {
        self.layer(layer).biases().as_slice()
    }

    /// Sets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    pub fn set_weight(&mut self, layer: usize, neuron: usize, input: usize, weight: f64) {
        self.layer_mut(layer).set_weight(neuron, input, weight);
    }

    /// Returns the weight from neuron `input` of layer `layer - 1` to neuron `neuron` of
    /// layer `layer`.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0, since the input layer has no weights, or past the output
    /// layer, or if `neuron` or `input` is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let mut nn = NeuralNetwork::<2, 1>::new(&[3]);
    /// nn.set_weight(2, 0, 1, 0.5);
    ///
    /// assert_eq!(nn.weight(2, 0, 1), 0.5);
    /// ```
    pub fn weight(&self, layer: usize, neuron: usize, input: usize) -> f64 {
        self.layer(layer).weights()[(neuron, input)]
    }

    /// Gets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    #[deprecated(since = "0.5.0", note = "renamed to `weight`")]
    pub fn get_weight(&self, layer: usize, neuron: usize, input: usize) -> f64 {
        self.weight(layer, neuron, input)
    }

    /// Sets the bias of a specific neuron. The layer index must be greater than 0 since
    /// the input layer does not have biases.
    pub fn set_bias(&mut self, layer: usize, neuron: usize, bias: f64) {
        self.layer_mut(layer).set_bias(neuron, bias);
    }

    /// Returns the bias of neuron `neuron` of layer `layer`.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0, since the input layer has no biases, or past the output
    /// layer, or if `neuron` is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let mut nn = NeuralNetwork::<2, 1>::new(&[3]);
    /// nn.set_bias(1, 2, -0.1);
    ///
    /// assert_eq!(nn.bias(1, 2), -0.1);
    /// ```
    pub fn bias(&self, layer: usize, neuron: usize) -> f64 {
        self.layer(layer).biases()[neuron]
    }

    /// Gets the bias of a specific neuron. The layer index must be greater than 0 since
    /// the input layer does not have biases.
    #[deprecated(since = "0.5.0", note = "renamed to `bias`")]
    pub fn get_bias(&self, layer: usize, neuron: usize) -> f64 {
        self.bias(layer, neuron)
    }

    /// The layer that receives weights for `layer`, which counts the input layer as 0.
    fn layer(&self, layer: usize) -> &Layer {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        &self.layers[layer - 1]
    }

    /// Every layer after the input layer, for the model writer.
    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    fn layer_mut(&mut self, layer: usize) -> &mut Layer {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        &mut self.layers[layer - 1]
    }

    /// Returns the number of layers of the neural network, counting the input layer.
    pub fn num_layers(&self) -> usize {
        self.layers.len() + 1
    }

    /// Returns the number of neurons of the given layer. Layer 0 is the input layer.
    pub fn layer_size(&self, layer: usize) -> usize {
        if layer == 0 {
            return IN;
        }
        self.layer(layer).size()
    }

    /// Returns the sizes of the hidden layers, the same list given to
    /// [`new`](Self::new) or [`from_parameters`](Self::from_parameters).
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let nn = NeuralNetwork::<4, 2>::new(&[8, 6]);
    ///
    /// let copy = NeuralNetwork::<4, 2>::from_parameters(&nn.hidden_layer_sizes(), &nn.parameters());
    /// assert_eq!(copy.hidden_layer_sizes(), vec![8, 6]);
    /// ```
    pub fn hidden_layer_sizes(&self) -> Vec<usize> {
        let hidden = &self.layers[..self.layers.len() - 1];
        hidden.iter().map(Layer::size).collect()
    }

    /// Returns the activation function of the given layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0, since the input layer has no activation, or past the
    /// output layer.
    pub fn layer_activation(&self, layer: usize) -> ActivationFunction {
        self.layer(layer).activation()
    }

    /// Sets the activation function of the given layer.
    ///
    /// # Panics
    ///
    /// Panics if `layer` is 0, since the input layer has no activation, or past the
    /// output layer.
    pub fn set_layer_activation(&mut self, layer: usize, activation_function: ActivationFunction) {
        self.layer_mut(layer).set_activation(activation_function);
    }

    /// Sets the activation function of every layer of the network.
    ///
    /// Combine it with [`set_output_activation`](Self::set_output_activation) to give
    /// the hidden layers and the output layer different functions.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, NeuralNetwork};
    /// let mut nn = NeuralNetwork::<2, 1>::new(&[3]);
    /// nn.set_activation_function(ActivationFunction::ReLU);
    /// nn.set_output_activation(ActivationFunction::Tanh);
    ///
    /// assert_eq!(nn.layer_activation(1), ActivationFunction::ReLU);
    /// assert_eq!(nn.layer_activation(2), ActivationFunction::Tanh);
    /// ```
    pub fn set_activation_function(&mut self, activation_function: ActivationFunction) {
        for layer in &mut self.layers {
            layer.set_activation(activation_function);
        }
    }

    /// Sets the activation function of the output layer only.
    pub fn set_output_activation(&mut self, activation_function: ActivationFunction) {
        self.set_layer_activation(self.num_layers() - 1, activation_function);
    }

    /// Returns the activation function of the output layer.
    pub fn output_activation(&self) -> ActivationFunction {
        self.layer_activation(self.num_layers() - 1)
    }
}

impl<const IN: usize, const OUT: usize> From<NeuralNetwork<IN, OUT>> for NetworkData {
    fn from(network: NeuralNetwork<IN, OUT>) -> Self {
        NetworkData::from(&network)
    }
}

impl<const IN: usize, const OUT: usize> From<&NeuralNetwork<IN, OUT>> for NetworkData {
    fn from(network: &NeuralNetwork<IN, OUT>) -> Self {
        let layers = network
            .layers
            .iter()
            .map(|layer| LayerData {
                activation: layer.activation(),
                weights: layer.weight_rows(),
                biases: layer.biases().iter().copied().collect(),
            })
            .collect();

        NetworkData { layers }
    }
}

impl<const IN: usize, const OUT: usize> TryFrom<NetworkData> for NeuralNetwork<IN, OUT> {
    type Error = ModelError;

    /// Checks that the stored layers chain from `IN` inputs to `OUT` outputs and that
    /// each layer agrees with itself.
    fn try_from(data: NetworkData) -> Result<Self, Self::Error> {
        let first = data.layers.first().ok_or(ModelError::EmptyNetwork)?;
        if let Some(row) = first.weights.first() {
            if row.len() != IN {
                return Err(ModelError::DimensionMismatch {
                    end: "input",
                    expected: IN,
                    found: row.len(),
                });
            }
        }

        let mut inputs = IN;
        let mut layers = Vec::with_capacity(data.layers.len());
        for (index, layer) in data.layers.iter().enumerate() {
            let layer = Layer::from_parts(&layer.weights, &layer.biases, layer.activation, inputs)
                .map_err(|reason| ModelError::InconsistentLayer {
                    layer: index + 1,
                    reason,
                })?;
            inputs = layer.size();
            layers.push(layer);
        }

        if inputs != OUT {
            return Err(ModelError::DimensionMismatch {
                end: "output",
                expected: OUT,
                found: inputs,
            });
        }

        Ok(Self { layers })
    }
}

impl<const IN: usize, const OUT: usize> fmt::Display for NeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Neural Network")?;
        writeln!(f)?;
        writeln!(f, "Input Layer Size: {IN}")?;
        writeln!(f)?;
        for (index, layer) in self.layers.iter().enumerate() {
            writeln!(f, "Layer {}: {}", index + 1, layer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation_functions::{binary_step, identity, relu, sigmoid, tanh};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const EPSILON: f64 = 1e-12;

    fn assert_all_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < EPSILON, "at index {i}: expected {e}, got {a}");
        }
    }

    /// A 2 -> 1 network with known weights, so outputs can be checked by hand.
    fn fixed_network() -> NeuralNetwork<2, 1> {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(1, &[[0.5, -0.25]]);
        nn.set_layer_biases(1, &[0.1]);
        nn
    }

    #[test]
    fn new_reports_layer_count_and_sizes() {
        let nn = NeuralNetwork::<3, 2>::new(&[5]);

        assert_eq!(nn.num_layers(), 3);
        assert_eq!(nn.layer_size(0), 3);
        assert_eq!(nn.layer_size(1), 5);
        assert_eq!(nn.layer_size(2), 2);
    }

    /// A network with no hidden layers is still a valid two-layer network. The old
    /// "fewer than two layers" runtime check is gone because IN and OUT guarantee it.
    #[test]
    fn new_without_hidden_layers_wires_input_straight_to_output() {
        let nn = NeuralNetwork::<3, 2>::new(&[]);

        assert_eq!(nn.num_layers(), 2);
        assert_eq!(nn.layer_size(0), 3);
        assert_eq!(nn.layer_size(1), 2);
    }

    #[test]
    #[should_panic(expected = "at least one neuron")]
    fn new_rejects_a_zero_sized_hidden_layer() {
        NeuralNetwork::<2, 1>::new(&[0]);
    }

    #[test]
    fn new_with_rng_is_reproducible_for_the_same_seed() {
        use rand::SeedableRng;

        let mut first_rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut second_rng = rand::rngs::StdRng::seed_from_u64(42);

        let first = NeuralNetwork::<3, 2>::new_with_rng(&[4], &mut first_rng);
        let second = NeuralNetwork::<3, 2>::new_with_rng(&[4], &mut second_rng);

        for layer in 1..first.num_layers() {
            for neuron in 0..first.layer_size(layer) {
                for input in 0..first.layer_size(layer - 1) {
                    assert_eq!(
                        first.weight(layer, neuron, input),
                        second.weight(layer, neuron, input),
                        "weight differs at layer {layer}, neuron {neuron}, input {input}"
                    );
                }
            }
        }
    }

    #[test]
    fn feed_forward_applies_weights_bias_and_activation() {
        let nn = fixed_network();

        // 0.5 * 1.0 + (-0.25) * 2.0 + 0.1 = 0.1
        let output = nn.feed_forward(&[1.0, 2.0]);

        assert_all_close(&output, &[sigmoid(0.1)]);
    }

    #[test]
    fn feed_forward_batch_matches_feed_forward_for_every_input() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut nn = NeuralNetwork::<3, 2>::new_with_rng(&[5, 4], &mut rng);
        nn.set_layer_activation(1, ActivationFunction::ReLU);
        nn.set_layer_activation(2, ActivationFunction::Tanh);

        let inputs: Vec<[f64; 3]> = (0..7).map(|i| [i as f64 * 0.3 - 1.0, 0.5, -(i as f64)]).collect();

        let batch = nn.feed_forward_batch(&inputs);

        assert_eq!(batch.len(), inputs.len());
        for (input, output) in inputs.iter().zip(&batch) {
            assert_all_close(output, &nn.feed_forward(input));
        }
    }

    #[test]
    fn feed_forward_batch_of_nothing_is_empty() {
        let nn = NeuralNetwork::<2, 1>::new(&[2]);

        assert!(nn.feed_forward_batch(&[]).is_empty());
    }

    #[test]
    fn feed_forward_batch_works_without_hidden_layers() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(1, &[[0.5, -0.25]]);
        nn.set_layer_biases(1, &[0.1]);
        nn.set_output_activation(ActivationFunction::Identity);

        let outputs = nn.feed_forward_batch(&[[1.0, 1.0], [2.0, 0.0]]);

        assert_all_close(&outputs.concat(), &[0.35, 1.1]);
    }

    #[test]
    fn feed_forward_defaults_to_sigmoid() {
        let nn = fixed_network();

        assert_eq!(nn.layer_activation(1), ActivationFunction::Sigmoid);
        assert_all_close(&nn.feed_forward(&[1.0, 2.0]), &[sigmoid(0.1)]);
    }

    /// The activation function used to be a field with no setter, so every
    /// network silently ran sigmoid regardless of what was configured.
    #[test]
    fn feed_forward_honours_every_activation_function() {
        let cases = [
            (ActivationFunction::Sigmoid, sigmoid as fn(f64) -> f64),
            (ActivationFunction::Tanh, tanh),
            (ActivationFunction::ReLU, relu),
            (ActivationFunction::BinaryStep, binary_step),
            (ActivationFunction::Identity, identity),
        ];

        for (variant, expected) in cases {
            let mut nn = fixed_network();
            nn.set_activation_function(variant);

            assert_eq!(nn.layer_activation(1), variant);
            assert_all_close(&nn.feed_forward(&[1.0, 2.0]), &[expected(0.1)]);
        }
    }

    /// `BinaryStep` was missing from the old lookup table and panicked here.
    #[test]
    fn binary_step_does_not_panic() {
        let mut nn = fixed_network();
        nn.set_activation_function(ActivationFunction::BinaryStep);

        assert_all_close(&nn.feed_forward(&[1.0, 2.0]), &[1.0]);
        assert_all_close(&nn.feed_forward(&[-1.0, 2.0]), &[0.0]);
    }

    #[test]
    fn set_weight_and_weight_round_trip() {
        let mut nn = NeuralNetwork::<2, 2>::new(&[]);
        nn.set_weight(1, 1, 0, 0.75);

        assert_eq!(nn.weight(1, 1, 0), 0.75);
    }

    #[test]
    #[should_panic(expected = "Invalid layer index")]
    fn set_layer_weights_rejects_layer_zero() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(0, &[[0.5, 0.5]]);
    }

    #[test]
    #[should_panic(expected = "Incompatible weights matrix size")]
    fn set_layer_weights_rejects_a_mismatched_matrix() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(1, &[[0.5, 0.5, 0.5]]);
    }

    #[test]
    #[should_panic(expected = "Incompatible biases vector size")]
    fn set_layer_biases_rejects_a_mismatched_vector() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_biases(1, &[0.1, 0.2]);
    }

    /// `Display` used to print the address of a function pointer here.
    #[test]
    fn display_names_the_activation_function() {
        let mut nn = fixed_network();
        nn.set_activation_function(ActivationFunction::ReLU);

        let rendered = nn.to_string();

        assert!(
            rendered.contains("Activation Function: ReLU"),
            "unexpected output:\n{rendered}"
        );
        assert!(
            !rendered.contains("0x"),
            "output leaked a pointer address:\n{rendered}"
        );
    }

    #[test]
    fn display_reports_the_input_layer_size() {
        let nn = NeuralNetwork::<4, 1>::new(&[2]);

        assert!(nn.to_string().contains("Input Layer Size: 4"));
    }

    #[test]
    fn layer_weights_and_biases_round_trip_in_row_major_order() {
        let mut nn = NeuralNetwork::<2, 3>::new(&[]);
        let weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];

        nn.set_layer_weights(1, &weights);
        nn.set_layer_biases(1, &[0.7, 0.8, 0.9]);

        assert_eq!(nn.layer_weights(1), weights.map(Vec::from).to_vec());
        assert_eq!(nn.layer_biases(1), &[0.7, 0.8, 0.9]);
        // Row = neuron, column = input, matching set_weight and weight.
        assert_eq!(nn.weight(1, 2, 0), 0.5);
    }

    #[test]
    fn set_layer_weights_accepts_weights_built_at_runtime() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        let weights: Vec<Vec<f64>> = vec![vec![0.5, -0.25]];

        nn.set_layer_weights(1, &weights);

        assert_eq!(nn.layer_weights(1), weights);
    }

    #[test]
    #[should_panic(expected = "Incompatible weights matrix size")]
    fn set_layer_weights_rejects_ragged_rows() {
        let mut nn = NeuralNetwork::<2, 2>::new(&[]);
        let ragged: Vec<Vec<f64>> = vec![vec![0.1, 0.2], vec![0.3]];

        nn.set_layer_weights(1, &ragged);
    }

    #[test]
    #[should_panic(expected = "Incompatible weights matrix size")]
    fn set_layer_weights_rejects_the_wrong_number_of_rows() {
        let mut nn = NeuralNetwork::<2, 2>::new(&[]);
        nn.set_layer_weights(1, &[[0.1, 0.2]]);
    }

    #[test]
    fn set_bias_and_bias_round_trip() {
        let mut nn = NeuralNetwork::<2, 2>::new(&[]);
        nn.set_bias(1, 1, -0.3);

        assert_eq!(nn.bias(1, 1), -0.3);
        assert_eq!(nn.layer_biases(1), &[0.0, -0.3]);
    }

    #[test]
    #[allow(deprecated)]
    fn the_deprecated_getters_still_read_the_same_values() {
        let nn = numbered_network();

        assert_eq!(nn.get_weight(1, 1, 0), nn.weight(1, 1, 0));
        assert_eq!(nn.get_bias(2, 0), nn.bias(2, 0));
    }

    #[test]
    #[should_panic(expected = "Invalid layer index")]
    fn layer_biases_rejects_layer_zero() {
        let nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.layer_biases(0);
    }

    /// A 2 -> 2 -> 1 network whose every parameter is distinct, so a wrong order shows.
    fn numbered_network() -> NeuralNetwork<2, 1> {
        let mut nn = NeuralNetwork::<2, 1>::new(&[2]);
        nn.set_layer_weights(1, &[[1.0, 2.0],
                                  [3.0, 4.0]]);
        nn.set_layer_biases(1, &[5.0, 6.0]);
        nn.set_layer_weights(2, &[[7.0, 8.0]]);
        nn.set_layer_biases(2, &[9.0]);
        nn
    }

    #[test]
    fn hidden_and_output_layers_can_use_different_activations() {
        let mut nn = NeuralNetwork::<1, 1>::new(&[1]);
        nn.set_layer_weights(1, &[[1.0]]);
        nn.set_layer_weights(2, &[[1.0]]);
        nn.set_activation_function(ActivationFunction::ReLU);
        nn.set_output_activation(ActivationFunction::Tanh);

        // relu(-2) = 0, then tanh(0) = 0; relu(2) = 2, then tanh(2).
        assert_all_close(&nn.feed_forward(&[-2.0]), &[0.0]);
        assert_all_close(&nn.feed_forward(&[2.0]), &[tanh(2.0)]);
        assert_eq!(nn.layer_activation(1), ActivationFunction::ReLU);
        assert_eq!(nn.output_activation(), ActivationFunction::Tanh);
    }

    #[test]
    fn set_layer_activation_changes_only_that_layer() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[3, 3]);
        nn.set_layer_activation(2, ActivationFunction::Identity);

        assert_eq!(nn.layer_activation(1), ActivationFunction::Sigmoid);
        assert_eq!(nn.layer_activation(2), ActivationFunction::Identity);
        assert_eq!(nn.layer_activation(3), ActivationFunction::Sigmoid);
    }

    #[test]
    #[should_panic(expected = "Invalid layer index")]
    fn the_input_layer_has_no_activation() {
        NeuralNetwork::<2, 1>::new(&[]).layer_activation(0);
    }

    #[test]
    fn parameters_follow_the_documented_order() {
        let nn = numbered_network();

        assert_eq!(
            nn.parameters(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn parameter_count_matches_the_shape() {
        let nn = NeuralNetwork::<10, 3>::new(&[8]);

        assert_eq!(nn.parameter_count(), 11 * 8 + 9 * 3);
        assert_eq!(nn.parameters().len(), nn.parameter_count());
        assert_eq!(NeuralNetwork::<10, 3>::parameter_count_for(&[8]), nn.parameter_count());
        assert_eq!(NeuralNetwork::<10, 3>::parameter_count_for(&[]), 11 * 3);
    }

    #[test]
    fn set_parameters_writes_the_documented_order() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[2]);
        nn.set_parameters(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        assert_eq!(nn, numbered_network());
        assert_eq!(nn.weight(1, 1, 0), 3.0);
        assert_eq!(nn.bias(2, 0), 9.0);
    }

    #[test]
    fn set_parameters_keeps_activation_functions() {
        let mut nn = numbered_network();
        nn.set_output_activation(ActivationFunction::Tanh);

        nn.set_parameters(&[0.0; 9]);

        assert_eq!(nn.output_activation(), ActivationFunction::Tanh);
    }

    #[test]
    fn parameters_round_trip_through_from_parameters() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let original = NeuralNetwork::<3, 2>::new_with_rng(&[4, 5], &mut rng);

        let rebuilt = NeuralNetwork::<3, 2>::from_parameters(
            &original.hidden_layer_sizes(),
            &original.parameters(),
        );

        assert_eq!(rebuilt, original);
        assert_all_close(
            &rebuilt.feed_forward(&[0.1, -0.2, 0.3]),
            &original.feed_forward(&[0.1, -0.2, 0.3]),
        );
    }

    #[test]
    #[should_panic(expected = "expected 9 parameters, got 8")]
    fn set_parameters_rejects_a_short_list() {
        numbered_network().set_parameters(&[0.0; 8]);
    }

    #[test]
    #[should_panic(expected = "expected 9 parameters, got 10")]
    fn from_parameters_rejects_a_long_list() {
        NeuralNetwork::<2, 1>::from_parameters(&[2], &[0.0; 10]);
    }

    #[test]
    #[should_panic(expected = "at least one neuron")]
    fn parameter_count_for_rejects_a_zero_sized_hidden_layer() {
        NeuralNetwork::<2, 1>::parameter_count_for(&[3, 0]);
    }

    #[test]
    fn hidden_layer_sizes_lists_only_hidden_layers() {
        assert_eq!(NeuralNetwork::<2, 1>::new(&[]).hidden_layer_sizes(), Vec::<usize>::new());
        assert_eq!(NeuralNetwork::<2, 1>::new(&[5, 3]).hidden_layer_sizes(), vec![5, 3]);
    }

    /// Fitness functions run in parallel (rayon), which needs Send + Sync.
    #[test]
    fn networks_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NeuralNetwork<10, 3>>();
    }
}

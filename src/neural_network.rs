use crate::activation_functions::{get_activation_function, ActivationFunction};
use crate::layer::Layer;
use crate::ModelError;
use nalgebra::{DMatrix, DVector};
use rand::{rng, Rng};
use std::fmt;
use serde::{Deserialize, Serialize};

/// Neural Network
///
/// This is the main struct of the library. It contains a vector of layers and an
/// activation function. You can use this struct and its methods to create, manipulate and
/// even implement your ways to train a neural network.
///
/// The number of input neurons (`IN`) and output neurons (`OUT`) are part of the type,
/// so feeding a wrongly sized input is a compile error rather than a runtime panic. The
/// hidden layers stay dynamic and are given at construction time.
///
/// # Example
///
/// ```
/// use only_brain::NeuralNetwork;
/// use nalgebra::dmatrix;
/// use nalgebra::dvector;
///
/// // A 2 -> 2 -> 1 network.
/// let mut nn = NeuralNetwork::<2, 1>::new(&[2]);
///
/// nn.set_layer_weights(1, dmatrix![0.1, 0.2;
///                                  0.3, 0.4]);
/// nn.set_layer_biases(1, dvector![0.1, 0.2]);
///
/// nn.set_layer_weights(2, dmatrix![0.9, 0.8]);
/// nn.set_layer_biases(2, dvector![0.1]);
///
/// let output = nn.feed_forward(&[0.5, 0.2]);
///
/// println!("{:?}", output);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "NetworkData")]
pub struct NeuralNetwork<const IN: usize, const OUT: usize> {
    layers: Vec<Layer>,
    activation_function: Option<ActivationFunction>,
}

/// The serialized form of a network, before its shape has been checked.
///
/// Deserializing goes through this so that every serde entry point, not only
/// [`crate::load_model`], rejects layers that do not fit together.
#[derive(Deserialize)]
pub(crate) struct NetworkData {
    layers: Vec<Layer>,
    activation_function: Option<ActivationFunction>,
}

impl<const IN: usize, const OUT: usize> NeuralNetwork<IN, OUT> {
    /// Creates a new Neural Network with the given hidden layer sizes. The input and
    /// output widths come from the type parameters, so `hidden` lists only the layers
    /// between them and may be empty.
    ///
    /// Weights are initialised at random; use [`NeuralNetwork::new_with_rng`] to control
    /// the seed.
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
        const { assert!(IN > 0, "a network needs at least one input neuron") };
        const { assert!(OUT > 0, "a network needs at least one output neuron") };
        assert!(
            hidden.iter().all(|&size| size > 0),
            "every hidden layer must have at least one neuron, got {hidden:?}"
        );

        let mut sizes = Vec::with_capacity(hidden.len() + 2);
        sizes.push(IN);
        sizes.extend_from_slice(hidden);
        sizes.push(OUT);

        let layers = sizes
            .iter()
            .zip(sizes.iter().skip(1))
            .map(|(a, b)| Layer::from_size(*b, *a, rng))
            .collect::<Vec<Layer>>();

        Self {
            layers,
            activation_function: None,
        }
    }

    /// Feeds the given inputs to the neural network and returns the output.
    ///
    /// The input and output widths are checked at compile time.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// # use nalgebra::dmatrix;
    /// # use nalgebra::dvector;
    /// let mut nn = NeuralNetwork::<1, 1>::new(&[]);
    ///
    /// nn.set_layer_weights(1, dmatrix![0.5]);
    /// nn.set_layer_biases(1, dvector![0.5]);
    ///
    /// let output = nn.feed_forward(&[0.5]);
    /// assert!((output[0] - 0.679178699175393).abs() < 1e-12);
    /// ```
    pub fn feed_forward(&self, inputs: &[f64; IN]) -> [f64; OUT] {
        let mut outputs = DVector::from_row_slice(inputs);
        let activation = get_activation_function(self.activation_function());

        for layer in &self.layers {
            outputs = layer.forward(&outputs, activation);
        }

        // The layer chain is built from IN..OUT and every setter checks its dimensions,
        // so the final vector always has exactly OUT elements.
        <[f64; OUT]>::try_from(outputs.as_slice())
            .expect("output layer width should match the OUT type parameter")
    }

    /// Sets the layer weights for the given layer. The weights matrix must have the size
    /// of the layer neurons x layer inputs. The layer index must be greater than 0 since it
    /// corresponds to the layer number that receives these weights.
    pub fn set_layer_weights(&mut self, layer: usize, weights: DMatrix<f64>) {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weights(weights);
    }

    /// Sets the layer biases for the given layer. The biases vector must have the size
    /// of the layer neurons. The layer index must be greater than 0 since the input layer
    /// does not have biases.
    pub fn set_layer_biases(&mut self, layer: usize, biases: DVector<f64>) {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_biases(biases);
    }

    /// Sets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    pub fn set_weight(&mut self, layer: usize, neuron: usize, input: usize, weight: f64) {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weight(neuron, input, weight);
    }

    /// Gets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    pub fn get_weight(&self, layer: usize, neuron: usize, input: usize) -> f64 {
        if layer == 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].weights()[(neuron, input)]
    }

    /// Returns the number of layers of the neural network.
    pub fn num_layers(&self) -> usize {
        self.layers.len() + 1
    }

    /// Returns the number of neurons of the given layer. Layer 0 is the input layer.
    pub fn layer_size(&self, layer: usize) -> usize {
        if layer == 0 {
            return IN;
        }
        self.layers[layer - 1].size()
    }

    /// Returns the activation function used by every layer of the network.
    ///
    /// Networks that have not had one set explicitly use
    /// [`ActivationFunction::Sigmoid`].
    pub fn activation_function(&self) -> ActivationFunction {
        self.activation_function.unwrap_or_default()
    }

    /// Sets the activation function applied by every layer of the network.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, NeuralNetwork};
    /// let mut nn = NeuralNetwork::<2, 1>::new(&[]);
    /// nn.set_activation_function(ActivationFunction::ReLU);
    ///
    /// assert_eq!(nn.activation_function(), ActivationFunction::ReLU);
    /// ```
    pub fn set_activation_function(&mut self, activation_function: ActivationFunction) {
        self.activation_function = Some(activation_function);
    }

    pub fn print(&self) {
        for layer in &self.layers {
            println!("{} {}", layer.weights(), layer.biases());
        }
    }
}

impl<const IN: usize, const OUT: usize> TryFrom<NetworkData> for NeuralNetwork<IN, OUT> {
    type Error = ModelError;

    /// Checks that the stored layers chain from `IN` inputs to `OUT` outputs and that
    /// each layer agrees with itself.
    fn try_from(data: NetworkData) -> Result<Self, Self::Error> {
        let first = data.layers.first().ok_or(ModelError::EmptyNetwork)?;
        if first.weights().ncols() != IN {
            return Err(ModelError::DimensionMismatch {
                end: "input",
                expected: IN,
                found: first.weights().ncols(),
            });
        }

        let mut inputs = IN;
        for (index, layer) in data.layers.iter().enumerate() {
            if let Some(reason) = layer.inconsistency(inputs) {
                return Err(ModelError::InconsistentLayer {
                    layer: index + 1,
                    reason,
                });
            }
            inputs = layer.size();
        }

        if inputs != OUT {
            return Err(ModelError::DimensionMismatch {
                end: "output",
                expected: OUT,
                found: inputs,
            });
        }

        Ok(Self {
            layers: data.layers,
            activation_function: data.activation_function,
        })
    }
}

impl<const IN: usize, const OUT: usize> fmt::Display for NeuralNetwork<IN, OUT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Neural Network")?;
        writeln!(f, "Activation Function: {:?}", self.activation_function())?;
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
    use crate::activation_functions::{binary_step, relu, sigmoid, tanh};
    use nalgebra::{dmatrix, dvector};

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
        nn.set_layer_weights(1, dmatrix![0.5, -0.25]);
        nn.set_layer_biases(1, dvector![0.1]);
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
                        first.get_weight(layer, neuron, input),
                        second.get_weight(layer, neuron, input),
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
    fn feed_forward_defaults_to_sigmoid() {
        let nn = fixed_network();

        assert_eq!(nn.activation_function(), ActivationFunction::Sigmoid);
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
        ];

        for (variant, expected) in cases {
            let mut nn = fixed_network();
            nn.set_activation_function(variant);

            assert_eq!(nn.activation_function(), variant);
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
    fn set_and_get_weight_round_trip() {
        let mut nn = NeuralNetwork::<2, 2>::new(&[]);
        nn.set_weight(1, 1, 0, 0.75);

        assert_eq!(nn.get_weight(1, 1, 0), 0.75);
    }

    #[test]
    #[should_panic(expected = "Invalid layer index")]
    fn set_layer_weights_rejects_layer_zero() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(0, dmatrix![0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Incompatible weights matrix size")]
    fn set_layer_weights_rejects_a_mismatched_matrix() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_weights(1, dmatrix![0.5, 0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Incompatible biases vector size")]
    fn set_layer_biases_rejects_a_mismatched_vector() {
        let mut nn = NeuralNetwork::<2, 1>::new(&[]);
        nn.set_layer_biases(1, dvector![0.1, 0.2]);
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
}

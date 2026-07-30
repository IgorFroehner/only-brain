use crate::activation_functions::{get_activation_function, ActivationFunction};
use crate::layer::Layer;
use nalgebra::{DMatrix, DVector};
use rand::rng;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Neural Network
///
/// This is the main struct of the library. It contains a vector of layers and an
/// activation function. You can use this struct and its methods to create, manipulate and
/// even implement your ways to train a neural network.
///
/// # Example
///
/// ```
/// use only_brain::NeuralNetwork;
/// use nalgebra::dmatrix;
/// use nalgebra::dvector;
///
/// fn main() {
///     let mut nn = NeuralNetwork::new(&vec![2, 2, 1]);
///
///     nn.set_layer_weights(1, dmatrix![0.1, 0.2;
///                                      0.3, 0.4]);
///     nn.set_layer_biases(1, dvector![0.1, 0.2]);
///
///     nn.set_layer_weights(2, dmatrix![0.9, 0.8]);
///     nn.set_layer_biases(2, dvector![0.1]);
///
///     let input = vec![0.5, 0.2];
///     let output = nn.feed_forward(&input);
///
///     println!("{:?}", output);
/// }
/// ```
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    activation_function: Option<ActivationFunction>,
}

impl NeuralNetwork {
    /// Creates a new Neural Network with the given layers. The layers vector must contain
    /// the number of neurons for each layer.
    ///
    /// # Panics
    ///
    /// Panics if fewer than two layer sizes are given (a network needs at least an input
    /// and an output layer), or if any layer size is zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// let nn = NeuralNetwork::new(&vec![2, 2, 1]);
    /// ```
    pub fn new(layers: &Vec<usize>) -> Self {
        assert!(
            layers.len() >= 2,
            "a neural network needs at least an input and an output layer, got {}",
            layers.len()
        );
        assert!(
            layers.iter().all(|&size| size > 0),
            "every layer must have at least one neuron, got {layers:?}"
        );

        let mut rng = rng();

        let layers = layers
            .iter()
            .zip(layers.iter().skip(1))
            .map(|(a, b)| Layer::from_size(*b, *a, &mut rng))
            .collect::<Vec<Layer>>();

        Self {
            layers,
            activation_function: None,
        }
    }

    /// Feeds the given inputs to the neural network and returns the output. The inputs
    /// vector must have the same size as the first layer of the network.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// # use nalgebra::dmatrix;
    /// # use nalgebra::dvector;
    /// # fn main() {
    /// let mut nn = NeuralNetwork::new(&vec![1, 1]);
    ///
    /// nn.set_layer_weights(1, dmatrix![0.5]);
    /// nn.set_layer_biases(1, dvector![0.5]);
    ///
    /// let input = vec![0.5];
    /// let output = nn.feed_forward(&input);
    /// assert_eq!(output, vec![0.679178699175393]);
    /// # }
    /// ```
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = DVector::from(Vec::clone(inputs));
        let activation = get_activation_function(self.activation_function());

        for layer in &self.layers {
            outputs = layer.forward(&outputs, activation);
        }

        outputs.data.into()
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

    /// Returns the number of neurons of the given layer.
    pub fn layer_size(&self, layer: usize) -> usize {
        if layer == 0 {
            return self.input_layer_size();
        }
        self.layers[layer - 1].size()
    }

    fn input_layer_size(&self) -> usize {
        self.layers[0].weights().ncols()
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
    /// let mut nn = NeuralNetwork::new(&vec![2, 1]);
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

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Neural Network")?;
        writeln!(f, "Activation Function: {:?}", self.activation_function())?;
        writeln!(f)?;
        writeln!(f, "Input Layer Size: {}", self.input_layer_size())?;
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
    fn fixed_network() -> NeuralNetwork {
        let mut nn = NeuralNetwork::new(&vec![2, 1]);
        nn.set_layer_weights(1, dmatrix![0.5, -0.25]);
        nn.set_layer_biases(1, dvector![0.1]);
        nn
    }

    #[test]
    fn new_reports_layer_count_and_sizes() {
        let nn = NeuralNetwork::new(&vec![3, 5, 2]);

        assert_eq!(nn.num_layers(), 3);
        assert_eq!(nn.layer_size(0), 3);
        assert_eq!(nn.layer_size(1), 5);
        assert_eq!(nn.layer_size(2), 2);
    }

    #[test]
    #[should_panic(expected = "at least an input and an output layer")]
    fn new_rejects_a_single_layer() {
        NeuralNetwork::new(&vec![3]);
    }

    #[test]
    #[should_panic(expected = "at least an input and an output layer")]
    fn new_rejects_an_empty_layer_list() {
        NeuralNetwork::new(&vec![]);
    }

    #[test]
    #[should_panic(expected = "at least one neuron")]
    fn new_rejects_a_zero_sized_layer() {
        NeuralNetwork::new(&vec![2, 0, 1]);
    }

    #[test]
    fn feed_forward_applies_weights_bias_and_activation() {
        let nn = fixed_network();

        // 0.5 * 1.0 + (-0.25) * 2.0 + 0.1 = 0.1
        let output = nn.feed_forward(&vec![1.0, 2.0]);

        assert_all_close(&output, &[sigmoid(0.1)]);
    }

    #[test]
    fn feed_forward_defaults_to_sigmoid() {
        let nn = fixed_network();

        assert_eq!(nn.activation_function(), ActivationFunction::Sigmoid);
        assert_all_close(&nn.feed_forward(&vec![1.0, 2.0]), &[sigmoid(0.1)]);
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
            assert_all_close(&nn.feed_forward(&vec![1.0, 2.0]), &[expected(0.1)]);
        }
    }

    /// `BinaryStep` was missing from the old lookup table and panicked here.
    #[test]
    fn binary_step_does_not_panic() {
        let mut nn = fixed_network();
        nn.set_activation_function(ActivationFunction::BinaryStep);

        assert_all_close(&nn.feed_forward(&vec![1.0, 2.0]), &[1.0]);
        assert_all_close(&nn.feed_forward(&vec![-1.0, 2.0]), &[0.0]);
    }

    #[test]
    fn set_and_get_weight_round_trip() {
        let mut nn = NeuralNetwork::new(&vec![2, 2]);
        nn.set_weight(1, 1, 0, 0.75);

        assert_eq!(nn.get_weight(1, 1, 0), 0.75);
    }

    #[test]
    #[should_panic(expected = "Invalid layer index")]
    fn set_layer_weights_rejects_layer_zero() {
        let mut nn = NeuralNetwork::new(&vec![2, 1]);
        nn.set_layer_weights(0, dmatrix![0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Incompatible weights matrix size")]
    fn set_layer_weights_rejects_a_mismatched_matrix() {
        let mut nn = NeuralNetwork::new(&vec![2, 1]);
        nn.set_layer_weights(1, dmatrix![0.5, 0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "Incompatible biases vector size")]
    fn set_layer_biases_rejects_a_mismatched_vector() {
        let mut nn = NeuralNetwork::new(&vec![2, 1]);
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
        let nn = NeuralNetwork::new(&vec![4, 2, 1]);

        assert!(nn.to_string().contains("Input Layer Size: 4"));
    }
}

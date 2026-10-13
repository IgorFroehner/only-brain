use crate::{ModelError, NeuralNetwork, activation_functions::ActivationFunction, init::Init};
use nalgebra::SVectorView;
use rand::Rng;
use rand::distr::Uniform;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Perceptron
///
/// A single neuron with `N` inputs: one weight per input, a bias and an
/// [`ActivationFunction`]. You can use this struct and its methods to create,
/// manipulate and even implement your own ways to train a perceptron (find the
/// weights). Inputs and weights are plain arrays, so their length is checked at
/// compile time.
///
/// # Example
///
/// ```
/// use only_brain::{ActivationFunction, Perceptron};
///
/// let mut perceptron = Perceptron::<2>::new(ActivationFunction::Sigmoid);
///
/// perceptron.set_weights([0.5, -0.2]);
/// perceptron.set_bias(0.3);
/// println!("{}", perceptron);
///
/// let output = perceptron.feed_forward(&[0.6, 0.4]);
/// println!("Output: {}", output);
/// ```
///
/// # Serialization
///
/// Perceptrons implement serde's `Serialize` and `Deserialize` through a plain form,
/// `{"activation", "weights", "bias"}`. Deserializing checks that there are exactly
/// `N` weights.
///
/// # Saving to a file
///
/// The model file format stores networks, and a perceptron is a network with no hidden
/// layers and one output. Convert it with [`NeuralNetwork::from`] and save that with
/// [`crate::dump_model`]; [`crate::load_model`] gives it back as a
/// `NeuralNetwork<N, 1>` with the same weights, bias and activation.
///
/// Its outputs agree with the perceptron's up to floating-point rounding, not bit for
/// bit: the network adds up the weighted inputs in a different order. A
/// [`BinaryStep`](ActivationFunction::BinaryStep) perceptron can therefore classify an
/// input that lies within rounding of its decision boundary differently from the
/// network.
///
/// ```no_run
/// use only_brain::{dump_model, load_model, ActivationFunction, NeuralNetwork, Perceptron};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut perceptron = Perceptron::<2>::new(ActivationFunction::BinaryStep);
/// perceptron.set_weights([1.0, 1.0]);
/// perceptron.set_bias(-1.5);
///
/// dump_model(&NeuralNetwork::from(perceptron), "and_gate.bin")?;
///
/// let and_gate: NeuralNetwork<2, 1> = load_model("and_gate.bin")?;
/// assert_eq!(and_gate.feed_forward(&[1.0, 1.0]), [1.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "PerceptronData", try_from = "PerceptronData")]
pub struct Perceptron<const N: usize> {
    weights: [f64; N],
    bias: f64,
    activation_function: ActivationFunction,
}

/// The plain form of a perceptron, before its number of weights has been checked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PerceptronData {
    activation: ActivationFunction,
    weights: Vec<f64>,
    bias: f64,
}

impl<const N: usize> Perceptron<N> {
    /// Creates a perceptron with zero weights and bias.
    pub fn new(activation_function: ActivationFunction) -> Self {
        Self {
            weights: [0.0; N],
            bias: 0.0,
            activation_function,
        }
    }

    /// Creates a perceptron with weights drawn uniformly from `[-1, 1)` by the given
    /// random number generator, and a zero bias.
    ///
    /// These are the weights [`NeuralNetwork::new_with_rng`] would give a network with
    /// `N` inputs, one output and no hidden layers from the same generator.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, Perceptron};
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let perceptron = Perceptron::<3>::new_with_rng(ActivationFunction::Tanh, &mut rng);
    ///
    /// assert!(perceptron.weights().iter().all(|w| (-1.0..1.0).contains(w)));
    /// assert_eq!(perceptron.bias(), 0.0);
    /// ```
    pub fn new_with_rng<R: Rng>(activation_function: ActivationFunction, rng: &mut R) -> Self {
        let (low, high) = Init::default().bounds(1, N);
        let uniform = Uniform::new(low, high).expect("the default range is valid");

        Self {
            weights: std::array::from_fn(|_| rng.sample(uniform)),
            bias: 0.0,
            activation_function,
        }
    }

    /// Returns the weights, one per input.
    pub fn weights(&self) -> [f64; N] {
        self.weights
    }

    /// Replaces the weights, one per input.
    pub fn set_weights(&mut self, weights: [f64; N]) {
        self.weights = weights;
    }

    /// Returns the bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Replaces the bias.
    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    /// Returns the activation function.
    pub fn activation_function(&self) -> ActivationFunction {
        self.activation_function
    }

    /// Replaces the activation function.
    pub fn set_activation_function(&mut self, activation_function: ActivationFunction) {
        self.activation_function = activation_function;
    }

    /// Applies the activation function to the weighted sum of `inputs` plus the bias.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, Perceptron};
    /// let mut perceptron = Perceptron::<2>::new(ActivationFunction::Identity);
    /// perceptron.set_weights([0.5, -0.25]);
    /// perceptron.set_bias(0.1);
    ///
    /// // 0.5 * 2 - 0.25 * 4 + 0.1
    /// assert!((perceptron.feed_forward(&[2.0, 4.0]) - 0.1).abs() < 1e-12);
    /// ```
    pub fn feed_forward(&self, inputs: &[f64; N]) -> f64 {
        let weights = SVectorView::<f64, N>::from_slice(&self.weights);
        let inputs = SVectorView::<f64, N>::from_slice(inputs);

        self.activation_function
            .apply(weights.dot(&inputs) + self.bias)
    }
}

impl<const N: usize> Default for Perceptron<N> {
    /// A perceptron with zero weights and bias and the default
    /// [`ActivationFunction`], sigmoid.
    fn default() -> Self {
        Self::new(ActivationFunction::default())
    }
}

impl<const N: usize> From<Perceptron<N>> for NeuralNetwork<N, 1> {
    /// A network with no hidden layers whose single output neuron has the perceptron's
    /// weights, bias and activation, so it can be saved with [`crate::dump_model`].
    ///
    /// The network adds up the weighted inputs in a different order, so its outputs
    /// agree with the perceptron's up to floating-point rounding rather than bit for
    /// bit. With [`BinaryStep`](ActivationFunction::BinaryStep), an input within
    /// rounding of the decision boundary can be classified differently.
    ///
    /// ```
    /// # use only_brain::{ActivationFunction, NeuralNetwork, Perceptron};
    /// let mut perceptron = Perceptron::<2>::new(ActivationFunction::Tanh);
    /// perceptron.set_weights([0.5, -0.25]);
    /// perceptron.set_bias(0.1);
    ///
    /// let network = NeuralNetwork::from(perceptron.clone());
    ///
    /// assert_eq!(network.parameters(), vec![0.5, -0.25, 0.1]);
    /// assert_eq!(network.output_activation(), ActivationFunction::Tanh);
    ///
    /// let [output] = network.feed_forward(&[1.0, 1.0]);
    /// assert!((output - perceptron.feed_forward(&[1.0, 1.0])).abs() < 1e-12);
    /// ```
    fn from(perceptron: Perceptron<N>) -> Self {
        // Flat parameter order: the weights of the one neuron, then its bias.
        let parameters: Vec<f64> = perceptron
            .weights
            .iter()
            .copied()
            .chain([perceptron.bias])
            .collect();

        let mut network = NeuralNetwork::from_parameters(&[], &parameters);
        network.set_activation_function(perceptron.activation_function);
        network
    }
}

impl<const N: usize> From<Perceptron<N>> for PerceptronData {
    fn from(perceptron: Perceptron<N>) -> Self {
        Self {
            activation: perceptron.activation_function,
            weights: perceptron.weights.to_vec(),
            bias: perceptron.bias,
        }
    }
}

impl<const N: usize> TryFrom<PerceptronData> for Perceptron<N> {
    type Error = ModelError;

    /// Checks that there is one weight per input.
    fn try_from(data: PerceptronData) -> Result<Self, Self::Error> {
        let found = data.weights.len();
        let weights =
            <[f64; N]>::try_from(data.weights).map_err(|_| ModelError::DimensionMismatch {
                end: "input",
                expected: N,
                found,
            })?;

        Ok(Self {
            weights,
            bias: data.bias,
            activation_function: data.activation,
        })
    }
}

impl<const N: usize> fmt::Display for Perceptron<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Perceptron:")?;
        writeln!(f, "Activation Function: {:?}", self.activation_function)?;
        writeln!(f)?;
        writeln!(f, "Inputs Size: {}", N)?;
        writeln!(f, "Weights: {:?}", self.weights)?;
        writeln!(f, "Bias: {}", self.bias)?;
        writeln!(f)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn a_new_perceptron_has_zero_weights_and_bias() {
        let perceptron = Perceptron::<3>::new(ActivationFunction::Identity);

        assert_eq!(perceptron.weights(), [0.0; 3]);
        assert_eq!(perceptron.bias(), 0.0);
        assert_eq!(perceptron.feed_forward(&[7.0, -2.0, 3.0]), 0.0);
    }

    #[test]
    fn feed_forward_applies_weights_bias_and_activation() {
        let mut perceptron = Perceptron::<2>::new(ActivationFunction::Identity);
        perceptron.set_weights([0.5, -0.25]);
        perceptron.set_bias(0.1);

        // 0.5 * 2 - 0.25 * 4 + 0.1
        assert!((perceptron.feed_forward(&[2.0, 4.0]) - 0.1).abs() < EPSILON);

        let mut perceptron = Perceptron::<2>::new(ActivationFunction::Sigmoid);
        perceptron.set_weights([0.5, -0.25]);
        perceptron.set_bias(0.1);
        let expected = 1.0 / (1.0 + (-0.1f64).exp());
        assert!((perceptron.feed_forward(&[2.0, 4.0]) - expected).abs() < EPSILON);
    }

    #[test]
    fn a_binary_step_perceptron_computes_or() {
        let mut or_gate = Perceptron::<2>::new(ActivationFunction::BinaryStep);
        or_gate.set_weights([1.0, 1.0]);
        or_gate.set_bias(-0.5);

        for (inputs, expected) in [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 1.0),
        ] {
            assert_eq!(or_gate.feed_forward(&inputs), expected, "OR{inputs:?}");
        }
    }

    #[test]
    fn getters_return_what_the_setters_stored() {
        let mut perceptron = Perceptron::<3>::default();
        perceptron.set_weights([0.1, 0.2, 0.3]);
        perceptron.set_bias(-0.4);
        perceptron.set_activation_function(ActivationFunction::ReLU);

        assert_eq!(perceptron.weights(), [0.1, 0.2, 0.3]);
        assert_eq!(perceptron.bias(), -0.4);
        assert_eq!(perceptron.activation_function(), ActivationFunction::ReLU);
    }

    #[test]
    fn the_default_perceptron_is_a_zero_sigmoid() {
        assert_eq!(
            Perceptron::<2>::default(),
            Perceptron::<2>::new(ActivationFunction::Sigmoid)
        );
    }

    #[test]
    fn new_with_rng_is_reproducible_and_matches_a_network_without_hidden_layers() {
        let first = Perceptron::<4>::new_with_rng(
            ActivationFunction::Sigmoid,
            &mut StdRng::seed_from_u64(5),
        );
        let second = Perceptron::<4>::new_with_rng(
            ActivationFunction::Sigmoid,
            &mut StdRng::seed_from_u64(5),
        );
        let network = NeuralNetwork::<4, 1>::new_with_rng(&[], &mut StdRng::seed_from_u64(5));

        assert_eq!(first, second);
        assert_eq!(NeuralNetwork::from(first), network);
    }

    #[test]
    fn a_perceptron_converts_to_an_equivalent_network() {
        let mut perceptron =
            Perceptron::<3>::new_with_rng(ActivationFunction::Tanh, &mut StdRng::seed_from_u64(11));
        perceptron.set_bias(0.2);

        let network = NeuralNetwork::from(perceptron.clone());

        assert_eq!(network.num_layers(), 2);
        assert_eq!(network.layer_activation(1), ActivationFunction::Tanh);
        assert_eq!(
            network.layer_weights(1),
            vec![perceptron.weights().to_vec()]
        );
        assert_eq!(network.layer_biases(1), &[0.2]);
        for inputs in [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [-0.3, 0.7, 4.0]] {
            let [output] = network.feed_forward(&inputs);
            assert!(
                (output - perceptron.feed_forward(&inputs)).abs() < EPSILON,
                "{inputs:?}"
            );
        }
    }

    /// Wide enough that the perceptron's `dot` and the network's `gemv` add up the
    /// terms in a different order, so the outputs may differ in the last bits.
    #[test]
    fn a_wide_perceptron_and_its_network_agree_up_to_rounding() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut perceptron = Perceptron::<64>::new_with_rng(ActivationFunction::Identity, &mut rng);
        perceptron.set_bias(0.3);
        let network = NeuralNetwork::from(perceptron.clone());

        for _ in 0..100 {
            let inputs: [f64; 64] = std::array::from_fn(|_| rng.random_range(-5.0..5.0));
            let [output] = network.feed_forward(&inputs);
            let expected = perceptron.feed_forward(&inputs);

            assert!((output - expected).abs() < 1e-10, "{output} vs {expected}");
        }
    }

    #[test]
    fn serde_round_trips_through_the_plain_form() {
        let mut perceptron = Perceptron::<2>::new(ActivationFunction::BinaryStep);
        perceptron.set_weights([0.5, -0.25]);
        perceptron.set_bias(0.1);

        let json = serde_json::to_string(&perceptron).unwrap();
        assert_eq!(
            json,
            r#"{"activation":"BinaryStep","weights":[0.5,-0.25],"bias":0.1}"#
        );

        let back: Perceptron<2> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, perceptron);
    }

    #[test]
    fn deserializing_too_many_weights_is_rejected() {
        let json = r#"{"activation":"Sigmoid","weights":[0.5,-0.25,1.0],"bias":0.1}"#;

        let error = serde_json::from_str::<Perceptron<2>>(json).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("3 input neurons but 2 were expected"),
            "{error}"
        );
    }

    #[test]
    fn deserializing_too_few_weights_is_rejected() {
        let json = r#"{"activation":"Sigmoid","weights":[0.5],"bias":0.1}"#;

        let error = serde_json::from_str::<Perceptron<2>>(json).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("1 input neurons but 2 were expected"),
            "{error}"
        );
    }

    #[test]
    fn display_shows_activation_size_weights_and_bias() {
        let mut perceptron = Perceptron::<4>::new(ActivationFunction::Tanh);
        perceptron.set_bias(0.75);

        let text = perceptron.to_string();

        assert!(text.contains("Activation Function: Tanh"), "{text}");
        assert!(text.contains("Inputs Size: 4"), "{text}");
        assert!(text.contains("Weights: [0.0, 0.0, 0.0, 0.0]"), "{text}");
        assert!(text.contains("Bias: 0.75"), "{text}");
    }

    #[test]
    fn perceptrons_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Perceptron<8>>();
    }
}

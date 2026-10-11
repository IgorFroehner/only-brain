use crate::{
    activation_functions::{get_activation_function, ActivationFunction},
    BVector,
};
use std::fmt;

/// Perceptron
///
/// This is a single perceptron implementation. It contains weights, bias and an activation function.
/// You can use this struct and its methods to create, manipulate and even implement your ways to
/// train a perceptron (find the weights).
/// # Example
/// ```
/// use only_brain::Perceptron;
/// use only_brain::ActivationFunction;
/// use only_brain::bvector;
///
/// fn main() {
///     let mut perceptron = Perceptron::<2>::new(ActivationFunction::Sigmoid);
///
///     perceptron.set_weights(bvector![0.5, -0.2]);
///     perceptron.set_bias(0.3);
///     println!("{}", perceptron);
///
///     let inputs = bvector![0.6, 0.4];
///     let output = perceptron.feed_forward(&inputs);
///     println!("Output: {}", output);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Perceptron<const N: usize> {
    weights: BVector<f64, N>,
    bias: f64,
    activation_function: ActivationFunction,
}

impl<const N: usize> Perceptron<N> {
    pub fn new(activation_function: ActivationFunction) -> Self {
        let weights = BVector::<f64, N>::from_element(0.0);
        let bias = 0.0;

        Self {
            weights,
            bias,
            activation_function,
        }
    }

    pub fn set_weights(&mut self, weights: BVector<f64, N>) {
        self.weights = weights;
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn feed_forward(&self, inputs: &BVector<f64, N>) -> f64 {
        let weighted_sum = self.weights.dot(inputs) + self.bias;
        get_activation_function(self.activation_function)(weighted_sum)
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
    use crate::bvector;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn a_new_perceptron_has_zero_weights_and_bias() {
        let perceptron = Perceptron::<3>::new(ActivationFunction::Identity);

        assert_eq!(perceptron.feed_forward(&bvector![7.0, -2.0, 3.0]), 0.0);
    }

    #[test]
    fn feed_forward_applies_weights_bias_and_activation() {
        let mut perceptron = Perceptron::<2>::new(ActivationFunction::Identity);
        perceptron.set_weights(bvector![0.5, -0.25]);
        perceptron.set_bias(0.1);

        // 0.5 * 2 - 0.25 * 4 + 0.1
        assert!((perceptron.feed_forward(&bvector![2.0, 4.0]) - 0.1).abs() < EPSILON);

        let mut perceptron = Perceptron::<2>::new(ActivationFunction::Sigmoid);
        perceptron.set_weights(bvector![0.5, -0.25]);
        perceptron.set_bias(0.1);
        let expected = 1.0 / (1.0 + (-0.1f64).exp());
        assert!((perceptron.feed_forward(&bvector![2.0, 4.0]) - expected).abs() < EPSILON);
    }

    #[test]
    fn a_binary_step_perceptron_computes_or() {
        let mut or_gate = Perceptron::<2>::new(ActivationFunction::BinaryStep);
        or_gate.set_weights(bvector![1.0, 1.0]);
        or_gate.set_bias(-0.5);

        for (inputs, expected) in [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 1.0)] {
            assert_eq!(or_gate.feed_forward(&BVector::from_array(inputs)), expected, "OR{inputs:?}");
        }
    }

    #[test]
    fn display_shows_activation_size_and_bias() {
        let mut perceptron = Perceptron::<4>::new(ActivationFunction::Tanh);
        perceptron.set_bias(0.75);

        let text = perceptron.to_string();

        assert!(text.contains("Activation Function: Tanh"), "{text}");
        assert!(text.contains("Inputs Size: 4"), "{text}");
        assert!(text.contains("Bias: 0.75"), "{text}");
    }
}

use crate::{
    activation_functions::{get_activation_function, ActivationFunction},
    BVector, LinAlgVector,
};
use std::fmt;

pub struct Perceptron<const N: usize> {
    weigths: BVector<f64, N>,
    bias: f64,
    activation_function: ActivationFunction,
}

impl<const N: usize> Perceptron<N> {
    pub fn new(activation_function: ActivationFunction) -> Self {
        let weigths = BVector::<f64, N>::from_element(0.0);
        let bias = 0.0;

        Self {
            weigths,
            bias,
            activation_function,
        }
    }

    pub fn set_weights(&mut self, weights: BVector<f64, N>) {
        self.weigths = weights;
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn feed_forward(&self, inputs: &BVector<f64, N>) -> f64 {
        let weighted_sum = self.weigths.dot(inputs) + self.bias;
        get_activation_function(self.activation_function)(weighted_sum)
    }
}

impl<const N: usize> fmt::Display for Perceptron<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Perceptron:")?;
        writeln!(f, "Activation Function: {:?}", self.activation_function)?;
        writeln!(f)?;
        writeln!(f, "Inputs Size: {}", N)?;
        writeln!(f, "Weights: {:?}", self.weigths)?;
        writeln!(f, "Bias: {}", self.bias)?;

        Ok(())
    }
}

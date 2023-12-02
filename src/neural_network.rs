use crate::activation_functions::{ActivationFunction, ACTION_FUNCTIONS_MAP};
use crate::layer::Layer;
use crate::math_utils::sigmoid;
use nalgebra::{DMatrix, DVector};
use rand::thread_rng;
use std::collections::HashMap;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    activation_function: Option<ActivationFunction>,
}

impl NeuralNetwork {
    pub fn new(layers: &Vec<usize>) -> Self {
        let mut rng = thread_rng();

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

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = DVector::from(Vec::clone(inputs));

        for layer in &self.layers {
            outputs = layer.forward(&outputs, self.activation_function());
        }

        outputs.data.into()
    }

    pub fn set_layer_weights(&mut self, layer: usize, weights: DMatrix<f64>) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weights(weights);
    }

    pub fn set_layer_biases(&mut self, layer: usize, biases: DVector<f64>) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_biases(biases);
    }

    pub fn set_weight(&mut self, layer: usize, neuron: usize, input: usize, weight: f64) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weight(neuron, input, weight);
    }

    pub fn activation_function(&self) -> fn(f64) -> f64 {
        if self.activation_function.is_none() {
            return sigmoid;
        }
        let functions_map = ACTION_FUNCTIONS_MAP
            .iter()
            .cloned()
            .collect::<HashMap<ActivationFunction, _>>();

        functions_map
            .get(&self.activation_function.unwrap())
            .unwrap()
            .clone()
    }

    pub fn print(&self) {
        for layer in &self.layers {
            println!("{} {}", layer.weights(), layer.biases());
        }
    }
}

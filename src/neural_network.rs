use crate::activation_functions::{sigmoid, ActivationFunction, ACTION_FUNCTIONS_MAP};
use crate::layer::Layer;
use nalgebra::{DMatrix, DVector};
use rand::thread_rng;
use std::collections::HashMap;

/// Neural Network
///
/// This is the main struct of the library. It contains a vector of layers and an
/// activation function. You can use this struct and its methods to create, manipulate and
/// even implement your ways to train a neural network.
///
/// # Example
///
/// ```
/// use no_brain::NeuralNetwork;
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
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    activation_function: Option<ActivationFunction>,
}

impl NeuralNetwork {
    /// Creates a new Neural Network with the given layers. The layers vector must contain
    /// the number of neurons for each layer.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::NeuralNetwork;
    /// # fn main() {
    /// let nn = NeuralNetwork::new(&vec![2, 2, 1]);
    /// # }
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

    /// Feeds the given inputs to the neural network and returns the output. The inputs
    /// vector must have the same size as the first layer of the network.
    ///
    /// # Example
    ///
    /// ```
    /// # use no_brain::NeuralNetwork;
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

        for layer in &self.layers {
            outputs = layer.forward(&outputs, self.activation_function());
        }

        outputs.data.into()
    }

    /// Sets the layer weights for the given layer. The weights matrix must have the size
    /// of the layer neurons x layer inputs. The layer index must be greater than 0 since it
    /// corresponds to the layer number that receives these weights.
    pub fn set_layer_weights(&mut self, layer: usize, weights: DMatrix<f64>) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weights(weights);
    }

    /// Sets the layer biases for the given layer. The biases vector must have the size
    /// of the layer neurons. The layer index must be greater than 0 since the input layer
    /// does not have biases.
    pub fn set_layer_biases(&mut self, layer: usize, biases: DVector<f64>) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_biases(biases);
    }

    /// Sets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    pub fn set_weight(&mut self, layer: usize, neuron: usize, input: usize, weight: f64) {
        if layer <= 0 {
            panic!("Invalid layer index");
        }
        self.layers[layer - 1].set_weight(neuron, input, weight);
    }

    /// Gets the weight of a specific neuron connection. The layer index must be greater
    /// than 0 since the input layer does not have weights.
    pub fn get_weight(&self, layer: usize, neuron: usize, input: usize) -> f64 {
        if layer <= 0 {
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
            return self.layers[0].weights().ncols();
        }
        self.layers[layer].size()
    }

    fn activation_function(&self) -> fn(f64) -> f64 {
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


//! # Only Brain
//!
//! A very simple Neural Network library built in Rust with the objective to allow
//! the user to create, manipulate and train a neural network directly. The user has
//! direct access to weights and biases of the network, allowing them to implement
//! their own training and manipulation methods.
//!
//! # Example
//!
//! The input and output widths are part of the network's type, so a wrongly sized
//! input is a compile error. Hidden layers are given at construction time, and weights
//! and biases are plain arrays, slices or `Vec`s.
//!
//! ```
//! use only_brain::NeuralNetwork;
//!
//! // A 2 -> 2 -> 1 network.
//! let mut nn = NeuralNetwork::<2, 1>::new(&[2]);
//!
//! // One row per neuron, one weight per neuron of the previous layer.
//! nn.set_layer_weights(1, &[[0.1, 0.2],
//!                           [0.3, 0.4]]);
//! nn.set_layer_biases(1, &[0.1, 0.2]);
//!
//! nn.set_layer_weights(2, &[[0.9, 0.8]]);
//! nn.set_layer_biases(2, &[0.1]);
//!
//! let output = nn.feed_forward(&[0.5, 0.2]);
//!
//! println!("{:?}", output);
//! ```
mod neural_network;
mod layer;
mod activation_functions;
mod perceptron;

mod io;
mod bvector;

pub use io::*;
pub use neural_network::*;
pub use perceptron::*;
pub use activation_functions::*;
pub use bvector::*;


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
//!
//! # Training it your way
//!
//! The library leaves learning to you. Two views of the same network make that easy:
//!
//! - **Per layer**: [`NeuralNetwork::layer_weights`], [`NeuralNetwork::set_layer_biases`]
//!   and friends, for methods that reason about layers, such as backpropagation.
//! - **Flat**: [`NeuralNetwork::parameters`], [`NeuralNetwork::set_parameters`] and
//!   [`NeuralNetwork::from_parameters`], for methods that search a list of numbers,
//!   such as genetic algorithms. A genome of
//!   [`NeuralNetwork::parameter_count_for`] genes is a network.
//!
//! ```
//! use only_brain::{ActivationFunction, NeuralNetwork};
//!
//! /// Scores a genome, as a genetic algorithm's fitness function would.
//! fn fitness(genome: &[f64]) -> f64 {
//!     let mut nn = NeuralNetwork::<2, 1>::from_parameters(&[3], genome);
//!     nn.set_activation_function(ActivationFunction::Tanh);
//!     nn.set_output_activation(ActivationFunction::Sigmoid);
//!
//!     let [output] = nn.feed_forward(&[1.0, 0.0]);
//!     -(1.0 - output).powi(2)
//! }
//!
//! let genome = vec![0.1; NeuralNetwork::<2, 1>::parameter_count_for(&[3])];
//! println!("fitness: {}", fitness(&genome));
//! ```
//!
//! Each layer has its own [`ActivationFunction`], and [`dump_model`] and
//! [`load_model`] save and load networks in a versioned format that keeps reading
//! files written by earlier versions.
mod neural_network;
mod layer;
mod activation_functions;
mod perceptron;

mod io;
mod bvector;
mod init;

pub use io::*;
pub use neural_network::*;
pub use perceptron::*;
pub use activation_functions::ActivationFunction;
#[allow(deprecated)]
pub use bvector::*;
pub use init::Init;


/// Compiles and runs the README's examples as doctests, so they cannot go stale.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

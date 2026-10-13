//! # Only Brain
//!
//! A small feed-forward neural network library for Rust, **without the learning part**.
//!
//! Only Brain gives you the network: fully connected layers, an activation function
//! per layer, a forward pass, and direct access to every weight and bias. How those
//! weights are found is up to you. Backpropagation, a genetic algorithm, random search
//! or a formula on a napkin all work, because the library never hides the numbers.
//!
//! - **Shape in the type.** [`NeuralNetwork<IN, OUT>`](NeuralNetwork) makes a wrongly
//!   sized input a compile error. Hidden layers are chosen at runtime.
//! - **Plain data at the boundary.** Weights and biases go in and out as arrays,
//!   slices and `Vec`s. You never touch a matrix type.
//! - **Two views of the same network.** Read and write it layer by layer, or as one
//!   flat list of parameters for black-box search.
//! - **Stable on disk.** A small, versioned binary format with no serialization
//!   dependency, plus serde for everything else. Old files keep loading.
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
//!   [`NeuralNetwork::feed_forward_trace`] runs a forward pass that keeps every
//!   layer's pre-activations and outputs in a [`Trace`], and
//!   [`ActivationFunction::derivative`] gives each activation's slope, which together
//!   are what backpropagation needs from the network.
//! - **Flat**: [`NeuralNetwork::parameters`], [`NeuralNetwork::set_parameters`] and
//!   [`NeuralNetwork::from_parameters`], for methods that search a list of numbers,
//!   such as genetic algorithms. A genome of
//!   [`NeuralNetwork::parameter_count_for`] genes is a network.
//!   [`NeuralNetwork::for_each_parameter_mut`] and [`NeuralNetwork::map_parameters`]
//!   visit the same parameters in place, in the same order, to mutate a network
//!   without copying its parameters out and back.
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
//! Each layer has its own [`ActivationFunction`]: sigmoid (the default), tanh, ReLU,
//! leaky ReLU, ELU, softplus, SiLU, binary step or identity. Softmax depends on the
//! whole layer, so it is not one of them: [`softmax`] and [`argmax`] turn an
//! `Identity` output layer's values into probabilities and a class.
//! [`NeuralNetwork::feed_forward_batch`] runs many inputs as one matrix product per
//! layer. [`dump_model`] and [`load_model`] save and load networks in a versioned
//! format that keeps reading files written by earlier versions.
//!
//! [`NeuralNetwork::new`] draws weights uniformly from `[-1, 1)`;
//! [`NeuralNetwork::new_with_init`] takes an [`Init`] that scales the range to each
//! layer instead, which deeper networks need.
//!
//! # Perceptron
//!
//! [`Perceptron`] is a single neuron over a plain array of `N` inputs. It converts into
//! a [`NeuralNetwork<N, 1>`](NeuralNetwork) with no hidden layers, which is how it is
//! saved.
//!
//! ```
//! use only_brain::{ActivationFunction, Perceptron};
//!
//! let mut or_gate = Perceptron::<2>::new(ActivationFunction::BinaryStep);
//! or_gate.set_weights([1.0, 1.0]);
//! or_gate.set_bias(-0.5);
//!
//! assert_eq!(or_gate.feed_forward(&[0.0, 1.0]), 1.0);
//! ```
#![warn(missing_docs)]

mod activation_functions;
mod layer;
mod neural_network;
mod perceptron;

mod bvector;
mod init;
mod io;

pub use activation_functions::{ActivationFunction, argmax, softmax};
#[allow(deprecated)]
pub use bvector::*;
pub use init::Init;
pub use io::*;
pub use neural_network::*;
pub use perceptron::*;

/// Compiles and runs the README's examples as doctests, so they cannot go stale.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

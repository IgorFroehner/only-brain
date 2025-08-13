use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    BinaryStep,
}

pub const ACTION_FUNCTIONS_MAP: [(ActivationFunction, fn(f64) -> f64); 3] = [
    (ActivationFunction::Sigmoid, sigmoid),
    (ActivationFunction::Tanh, tanh),
    (ActivationFunction::ReLU, relu),
];

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn binary_step(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub fn get_activation_function(func: ActivationFunction) -> fn(f64) -> f64 {
    match func {
        ActivationFunction::Sigmoid => sigmoid,
        ActivationFunction::Tanh => tanh,
        ActivationFunction::ReLU => relu,
        ActivationFunction::BinaryStep => binary_step,
    }
}


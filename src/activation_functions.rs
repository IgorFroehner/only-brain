use crate::math_utils::sigmoid;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ActivationFunction {
    Sigmoid,
    // Tanh,
    // ReLU,
    // LeakyReLU,
}

pub const ACTION_FUNCTIONS_MAP: [(ActivationFunction, fn(f64) -> f64); 1] = [
    (ActivationFunction::Sigmoid, sigmoid),
    // (ActivationFunction::Tanh, tanh),
    // (ActivationFunction::ReLU, relu),
    // (ActivationFunction::LeakyReLU, leaky_relu),
];

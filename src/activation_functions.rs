use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub enum ActivationFunction {
    #[default]
    Sigmoid,
    Tanh,
    ReLU,
    BinaryStep,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn sigmoid_is_one_half_at_zero_and_saturates_at_the_extremes() {
        assert_close(sigmoid(0.0), 0.5);
        assert_close(sigmoid(f64::INFINITY), 1.0);
        assert_close(sigmoid(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn relu_clamps_negatives_to_zero_and_passes_positives_through() {
        assert_close(relu(-3.5), 0.0);
        assert_close(relu(0.0), 0.0);
        assert_close(relu(2.25), 2.25);
    }

    #[test]
    fn binary_step_switches_at_zero_inclusive() {
        assert_close(binary_step(-0.001), 0.0);
        assert_close(binary_step(0.0), 1.0);
        assert_close(binary_step(0.001), 1.0);
    }

    #[test]
    fn tanh_is_odd_around_zero() {
        assert_close(tanh(0.0), 0.0);
        assert_close(tanh(1.3), -tanh(-1.3));
    }

    /// Guards the bug where a lookup table covered only 3 of the 4 variants and
    /// panicked on the missing one.
    #[test]
    fn every_variant_resolves_to_its_own_function() {
        let cases = [
            (ActivationFunction::Sigmoid, sigmoid as fn(f64) -> f64),
            (ActivationFunction::Tanh, tanh),
            (ActivationFunction::ReLU, relu),
            (ActivationFunction::BinaryStep, binary_step),
        ];

        for (variant, expected) in cases {
            let resolved = get_activation_function(variant);
            for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
                assert_close(resolved(x), expected(x));
            }
        }
    }

    #[test]
    fn default_activation_function_is_sigmoid() {
        assert_eq!(ActivationFunction::default(), ActivationFunction::Sigmoid);
    }
}


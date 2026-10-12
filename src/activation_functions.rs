use serde::{Deserialize, Serialize};

/// The function a layer applies to each neuron's weighted sum.
///
/// Every layer of a [`crate::NeuralNetwork`] has its own, so hidden layers and the
/// output layer can differ: a common choice is [`ReLU`](Self::ReLU) or
/// [`Tanh`](Self::Tanh) for hidden layers, with [`Tanh`](Self::Tanh) for outputs in
/// `[-1, 1]`, [`Sigmoid`](Self::Sigmoid) for outputs in `[0, 1]`, or
/// [`Identity`](Self::Identity) for unbounded outputs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ActivationFunction {
    /// `1 / (1 + e^-x)`, in `(0, 1)`.
    #[default]
    Sigmoid,
    /// `tanh(x)`, in `(-1, 1)`.
    Tanh,
    /// `max(x, 0)`.
    ReLU,
    /// `1` for `x >= 0`, else `0`.
    BinaryStep,
    /// `x`, unchanged.
    Identity,
}

impl ActivationFunction {
    /// Every activation function, in declaration order.
    pub const ALL: [ActivationFunction; 5] = [
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::ReLU,
        ActivationFunction::BinaryStep,
        ActivationFunction::Identity,
    ];

    /// Applies the function to `x`.
    ///
    /// ```
    /// # use only_brain::ActivationFunction;
    /// assert_eq!(ActivationFunction::ReLU.apply(-2.0), 0.0);
    /// assert_eq!(ActivationFunction::Identity.apply(-2.0), -2.0);
    /// ```
    pub fn apply(self, x: f64) -> f64 {
        get_activation_function(self)(x)
    }

    /// The byte that stands for this function in the model file format.
    ///
    /// These codes are part of the format and never change; new functions get new
    /// codes. The first four match the order of the format before 0.3.
    pub(crate) fn code(self) -> u8 {
        match self {
            ActivationFunction::Sigmoid => 0,
            ActivationFunction::Tanh => 1,
            ActivationFunction::ReLU => 2,
            ActivationFunction::BinaryStep => 3,
            ActivationFunction::Identity => 4,
        }
    }

    /// The inverse of [`ActivationFunction::code`].
    pub(crate) fn from_code(code: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|function| function.code() == code)
    }
}

pub(crate) fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub(crate) fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub(crate) fn binary_step(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub(crate) fn identity(x: f64) -> f64 {
    x
}

/// Returns the plain function behind an [`ActivationFunction`].
pub(crate) fn get_activation_function(func: ActivationFunction) -> fn(f64) -> f64 {
    match func {
        ActivationFunction::Sigmoid => sigmoid,
        ActivationFunction::Tanh => tanh,
        ActivationFunction::ReLU => relu,
        ActivationFunction::BinaryStep => binary_step,
        ActivationFunction::Identity => identity,
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
            (ActivationFunction::Identity, identity),
        ];
        assert_eq!(cases.len(), ActivationFunction::ALL.len());

        for (variant, expected) in cases {
            let resolved = get_activation_function(variant);
            for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
                assert_close(resolved(x), expected(x));
            }
        }
    }

    /// The codes are written into model files, so they must never move.
    #[test]
    fn file_format_codes_are_stable_and_round_trip() {
        let expected = [
            (ActivationFunction::Sigmoid, 0),
            (ActivationFunction::Tanh, 1),
            (ActivationFunction::ReLU, 2),
            (ActivationFunction::BinaryStep, 3),
            (ActivationFunction::Identity, 4),
        ];

        for (function, code) in expected {
            assert_eq!(function.code(), code);
            assert_eq!(ActivationFunction::from_code(code), Some(function));
        }
        assert_eq!(ActivationFunction::from_code(5), None);
    }

    #[test]
    fn identity_passes_values_through() {
        assert_close(identity(-3.25), -3.25);
        assert_close(ActivationFunction::Identity.apply(7.5), 7.5);
    }

    #[test]
    fn default_activation_function_is_sigmoid() {
        assert_eq!(ActivationFunction::default(), ActivationFunction::Sigmoid);
    }
}


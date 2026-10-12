use serde::{Deserialize, Serialize};

/// The function a layer applies to each neuron's weighted sum.
///
/// Every layer of a [`crate::NeuralNetwork`] has its own, so hidden layers and the
/// output layer can differ: a common choice is [`ReLU`](Self::ReLU) or
/// [`Tanh`](Self::Tanh) for hidden layers, with [`Tanh`](Self::Tanh) for outputs in
/// `[-1, 1]`, [`Sigmoid`](Self::Sigmoid) for outputs in `[0, 1]`, or
/// [`Identity`](Self::Identity) for unbounded outputs.
///
/// [`LeakyReLU`](Self::LeakyReLU), [`ELU`](Self::ELU), [`Softplus`](Self::Softplus)
/// and [`SiLU`](Self::SiLU) are smoother or never-flat alternatives to
/// [`ReLU`](Self::ReLU) for hidden layers: their slope for negative inputs is not
/// zero, so neurons pushed below zero can still recover. Their parameters are fixed,
/// because the model file stores a single byte per layer for its activation.
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
    /// `x` for `x >= 0`, else `0.01 · x`. The slope for negative inputs is fixed at
    /// `0.01`.
    LeakyReLU,
    /// Exponential linear unit with `α = 1`: `x` for `x >= 0`, else `e^x - 1`, in
    /// `(-1, ∞)`.
    ELU,
    /// `ln(1 + e^x)`, a smooth [`ReLU`](Self::ReLU), in `(0, ∞)`. Computed in a form
    /// that neither overflows for large `x` nor loses precision for very negative `x`.
    Softplus,
    /// Sigmoid linear unit, also called swish: `x · sigmoid(x)`, in about
    /// `[-0.278, ∞)`.
    SiLU,
}

impl ActivationFunction {
    /// Every activation function, in declaration order.
    pub const ALL: [ActivationFunction; 9] = [
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::ReLU,
        ActivationFunction::BinaryStep,
        ActivationFunction::Identity,
        ActivationFunction::LeakyReLU,
        ActivationFunction::ELU,
        ActivationFunction::Softplus,
        ActivationFunction::SiLU,
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

    /// The derivative of the function with respect to its input `x`, the neuron's
    /// pre-activation (weighted sum), as backpropagation needs it.
    ///
    /// Three functions are not differentiable everywhere, so they follow a convention:
    ///
    /// - [`ReLU`](Self::ReLU) has derivative `1` for `x > 0` and `0` for `x <= 0`,
    ///   so `0` at `x = 0`, the usual choice in deep learning libraries.
    /// - [`LeakyReLU`](Self::LeakyReLU) likewise has derivative `1` for `x > 0` and
    ///   `0.01` for `x <= 0`, so `0.01` at `x = 0`.
    /// - [`BinaryStep`](Self::BinaryStep) has derivative `0` everywhere, including at
    ///   `x = 0` where it jumps. No gradient flows through it, so a layer using it
    ///   cannot be trained by gradient descent; train it with a gradient-free method,
    ///   such as a genetic algorithm, instead.
    ///
    /// [`ELU`](Self::ELU) with `α = 1` has derivative `1` from both sides at `x = 0`,
    /// and [`Softplus`](Self::Softplus) and [`SiLU`](Self::SiLU) are smooth, so they
    /// need no convention.
    ///
    /// # Example
    ///
    /// ```
    /// # use only_brain::ActivationFunction;
    /// // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0))
    /// assert_eq!(ActivationFunction::Sigmoid.derivative(0.0), 0.25);
    /// assert_eq!(ActivationFunction::Tanh.derivative(0.0), 1.0);
    /// assert_eq!(ActivationFunction::ReLU.derivative(3.0), 1.0);
    /// assert_eq!(ActivationFunction::ReLU.derivative(0.0), 0.0);
    /// assert_eq!(ActivationFunction::BinaryStep.derivative(0.0), 0.0);
    /// assert_eq!(ActivationFunction::Identity.derivative(-2.0), 1.0);
    /// assert_eq!(ActivationFunction::LeakyReLU.derivative(-2.0), 0.01);
    /// assert_eq!(ActivationFunction::ELU.derivative(0.0), 1.0);
    /// // softplus'(x) = sigmoid(x)
    /// assert_eq!(ActivationFunction::Softplus.derivative(0.0), 0.5);
    /// // silu'(0) = sigmoid(0)
    /// assert_eq!(ActivationFunction::SiLU.derivative(0.0), 0.5);
    /// ```
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
            ActivationFunction::BinaryStep => 0.0,
            ActivationFunction::Identity => 1.0,
            ActivationFunction::LeakyReLU => {
                if x > 0.0 { 1.0 } else { LEAKY_RELU_SLOPE }
            }
            ActivationFunction::ELU => {
                if x >= 0.0 { 1.0 } else { x.exp() }
            }
            ActivationFunction::Softplus => sigmoid(x),
            ActivationFunction::SiLU => {
                let s = sigmoid(x);
                s * (1.0 + x * (1.0 - s))
            }
        }
    }

    /// The byte that stands for this function in the model file format.
    ///
    /// These codes are part of the format and never change; new functions get new
    /// codes. The first four match the order of the format before 0.3; 4 was added in
    /// 0.3 and 5 to 8 in 0.5.
    pub(crate) fn code(self) -> u8 {
        match self {
            ActivationFunction::Sigmoid => 0,
            ActivationFunction::Tanh => 1,
            ActivationFunction::ReLU => 2,
            ActivationFunction::BinaryStep => 3,
            ActivationFunction::Identity => 4,
            ActivationFunction::LeakyReLU => 5,
            ActivationFunction::ELU => 6,
            ActivationFunction::Softplus => 7,
            ActivationFunction::SiLU => 8,
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

/// The slope of [`ActivationFunction::LeakyReLU`] for negative inputs.
const LEAKY_RELU_SLOPE: f64 = 0.01;

pub(crate) fn leaky_relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { LEAKY_RELU_SLOPE * x }
}

pub(crate) fn elu(x: f64) -> f64 {
    if x >= 0.0 { x } else { x.exp_m1() }
}

/// `ln(1 + e^x)` rewritten as `max(x, 0) + ln(1 + e^-|x|)`, so `e^x` never overflows
/// for large `x`, and `ln_1p` keeps the tiny result exact for very negative `x`.
pub(crate) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

pub(crate) fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Returns the plain function behind an [`ActivationFunction`].
pub(crate) fn get_activation_function(func: ActivationFunction) -> fn(f64) -> f64 {
    match func {
        ActivationFunction::Sigmoid => sigmoid,
        ActivationFunction::Tanh => tanh,
        ActivationFunction::ReLU => relu,
        ActivationFunction::BinaryStep => binary_step,
        ActivationFunction::Identity => identity,
        ActivationFunction::LeakyReLU => leaky_relu,
        ActivationFunction::ELU => elu,
        ActivationFunction::Softplus => softplus,
        ActivationFunction::SiLU => silu,
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
            (ActivationFunction::LeakyReLU, leaky_relu),
            (ActivationFunction::ELU, elu),
            (ActivationFunction::Softplus, softplus),
            (ActivationFunction::SiLU, silu),
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
            (ActivationFunction::LeakyReLU, 5),
            (ActivationFunction::ELU, 6),
            (ActivationFunction::Softplus, 7),
            (ActivationFunction::SiLU, 8),
        ];
        assert_eq!(expected.len(), ActivationFunction::ALL.len());

        for (function, code) in expected {
            assert_eq!(function.code(), code);
            assert_eq!(ActivationFunction::from_code(code), Some(function));
        }
        assert_eq!(ActivationFunction::from_code(9), None);
    }

    /// Compares every derivative with a central finite difference of `apply`, away
    /// from zero, where ReLU, LeakyReLU and BinaryStep are not differentiable. The
    /// points close to zero check both sides of the ELU and LeakyReLU branches.
    #[test]
    fn every_derivative_matches_a_numeric_derivative() {
        const H: f64 = 1e-6;

        for function in ActivationFunction::ALL {
            for x in [-8.0, -3.0, -1.2, -0.4, -0.01, 0.01, 0.3, 0.9, 2.5, 8.0] {
                let numeric = (function.apply(x + H) - function.apply(x - H)) / (2.0 * H);
                let analytic = function.derivative(x);
                assert!(
                    (analytic - numeric).abs() < 1e-8,
                    "{function:?} at {x}: derivative {analytic}, numeric {numeric}"
                );
            }
        }
    }

    #[test]
    fn relu_derivative_is_zero_at_zero_and_one_just_above() {
        assert_eq!(ActivationFunction::ReLU.derivative(0.0), 0.0);
        assert_eq!(ActivationFunction::ReLU.derivative(-0.0), 0.0);
        assert_eq!(ActivationFunction::ReLU.derivative(f64::MIN_POSITIVE), 1.0);
    }

    #[test]
    fn leaky_relu_derivative_is_the_leak_at_zero_and_one_just_above() {
        assert_eq!(ActivationFunction::LeakyReLU.derivative(-0.0), 0.01);
        assert_eq!(ActivationFunction::LeakyReLU.derivative(0.0), 0.01);
        assert_eq!(ActivationFunction::LeakyReLU.derivative(f64::MIN_POSITIVE), 1.0);
    }

    /// With `α = 1` both branches of ELU have slope 1 at zero.
    #[test]
    fn elu_derivative_is_continuous_at_zero() {
        assert_eq!(ActivationFunction::ELU.derivative(0.0), 1.0);
        assert_eq!(ActivationFunction::ELU.derivative(-0.0), 1.0);
        assert_close(ActivationFunction::ELU.derivative(-f64::MIN_POSITIVE), 1.0);
    }

    #[test]
    fn binary_step_derivative_is_zero_everywhere_including_the_jump() {
        for x in [-1.0, -f64::MIN_POSITIVE, 0.0, f64::MIN_POSITIVE, 1.0] {
            assert_eq!(ActivationFunction::BinaryStep.derivative(x), 0.0);
        }
    }

    #[test]
    fn saturated_sigmoid_and_tanh_have_a_vanishing_derivative() {
        for function in [ActivationFunction::Sigmoid, ActivationFunction::Tanh] {
            assert_eq!(function.derivative(1000.0), 0.0);
            assert_eq!(function.derivative(-1000.0), 0.0);
        }
    }

    #[test]
    fn leaky_relu_scales_negatives_by_one_hundredth() {
        assert_close(leaky_relu(-3.0), -0.03);
        assert_close(leaky_relu(0.0), 0.0);
        assert_close(leaky_relu(2.25), 2.25);
    }

    #[test]
    fn elu_is_bounded_below_by_minus_one_and_linear_above_zero() {
        assert_close(elu(0.0), 0.0);
        assert_close(elu(2.25), 2.25);
        assert_close(elu(-1.0), (-1.0f64).exp() - 1.0);
        assert_eq!(elu(-1000.0), -1.0);
    }

    #[test]
    fn softplus_is_ln_two_at_zero_and_matches_the_naive_formula() {
        assert_close(softplus(0.0), 2f64.ln());
        for x in [-5.0, -1.0, 0.5, 3.0, 20.0] {
            assert_close(softplus(x), (1.0 + f64::exp(x)).ln());
        }
    }

    /// The naive `ln(1 + e^x)` returns infinity at `x = 1000`, because `e^1000`
    /// overflows, and 0 for very negative `x` long before the true value underflows.
    #[test]
    fn softplus_is_stable_for_large_inputs() {
        assert_eq!(softplus(1000.0), 1000.0);
        assert_eq!(softplus(-1000.0), 0.0);
        assert_eq!(softplus(-40.0), (-40.0f64).exp());
        for function in [ActivationFunction::Softplus, ActivationFunction::SiLU] {
            assert_eq!(function.derivative(1000.0), 1.0);
            assert_eq!(function.derivative(-1000.0), 0.0);
        }
    }

    #[test]
    fn silu_is_x_times_sigmoid() {
        assert_close(silu(0.0), 0.0);
        for x in [-3.0, -1.0, 0.5, 2.0] {
            assert_close(silu(x), x * sigmoid(x));
        }
        assert_eq!(silu(1000.0), 1000.0);
        assert_eq!(silu(-1000.0), 0.0);
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


/// How [`NeuralNetwork::new_with_init`](crate::NeuralNetwork::new_with_init) draws a
/// network's initial weights.
///
/// Every scheme draws each weight independently from a uniform distribution and sets
/// every bias to zero. They differ in the range, which the scaled schemes derive from
/// each layer's shape: its number of inputs (`fan_in`, the size of the previous layer)
/// and of neurons (`fan_out`).
///
/// A wide range saturates sigmoid and tanh in deeper networks: the weighted sums grow
/// with the number of inputs, the activations sit at their flat ends, and small changes
/// to the weights stop changing the output. [`Xavier`](Self::Xavier) keeps the sums in
/// a useful range for sigmoid and tanh layers, [`He`](Self::He) does the same for ReLU
/// layers.
///
/// # Example
///
/// ```
/// use only_brain::{Init, NeuralNetwork};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let nn = NeuralNetwork::<16, 4>::new_with_init(&[32, 32], Init::Xavier, &mut rng);
///
/// // Layer 1 has 16 inputs and 32 neurons: limit = sqrt(6 / (16 + 32)).
/// let limit = (6.0f64 / 48.0).sqrt();
/// assert!(nn.layer_weights(1).iter().flatten().all(|w| w.abs() <= limit));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Init {
    /// Uniform in `[low, high)` for every layer, whatever its shape. The default is
    /// `[-1, 1)`, which is what [`NeuralNetwork::new`](crate::NeuralNetwork::new) uses.
    Uniform {
        /// The smallest value a weight can take.
        low: f64,
        /// The bound no weight reaches.
        high: f64,
    },
    /// Xavier (Glorot) uniform: `[-limit, limit)` with
    /// `limit = sqrt(6 / (fan_in + fan_out))`. Suited to sigmoid and tanh layers.
    Xavier,
    /// He (Kaiming) uniform: `[-limit, limit)` with `limit = sqrt(6 / fan_in)`. Suited
    /// to ReLU layers.
    He,
}

impl Default for Init {
    /// `Uniform { low: -1.0, high: 1.0 }`, the initialisation of
    /// [`NeuralNetwork::new`](crate::NeuralNetwork::new).
    fn default() -> Self {
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        }
    }
}

impl Init {
    /// The `(low, high)` bounds for a layer of `neurons` neurons with `inputs` inputs.
    ///
    /// # Panics
    ///
    /// Panics if a [`Uniform`](Self::Uniform) range is empty or not finite, including
    /// when `high - low` overflows.
    pub(crate) fn bounds(self, neurons: usize, inputs: usize) -> (f64, f64) {
        match self {
            Init::Uniform { low, high } => {
                assert!(
                    low.is_finite() && high.is_finite() && low < high && (high - low).is_finite(),
                    "Init::Uniform needs finite bounds with low < high and a finite width, got [{low}, {high})"
                );
                (low, high)
            }
            Init::Xavier => Self::symmetric((6.0 / (inputs + neurons) as f64).sqrt()),
            Init::He => Self::symmetric((6.0 / inputs as f64).sqrt()),
        }
    }

    fn symmetric(limit: f64) -> (f64, f64) {
        (-limit, limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_range_new_has_always_used() {
        assert_eq!(
            Init::default(),
            Init::Uniform {
                low: -1.0,
                high: 1.0
            }
        );
        assert_eq!(Init::default().bounds(8, 3), (-1.0, 1.0));
    }

    #[test]
    fn xavier_scales_with_inputs_and_neurons() {
        assert_eq!(Init::Xavier.bounds(2, 4), (-1.0, 1.0));
        assert_eq!(Init::Xavier.bounds(16, 8), (-0.5, 0.5));
    }

    #[test]
    fn he_scales_with_inputs_only() {
        assert_eq!(Init::He.bounds(100, 6), (-1.0, 1.0));
        assert_eq!(Init::He.bounds(1, 24), (-0.5, 0.5));
    }

    #[test]
    #[should_panic(expected = "low < high")]
    fn an_empty_uniform_range_is_rejected() {
        Init::Uniform {
            low: 1.0,
            high: 1.0,
        }
        .bounds(1, 1);
    }

    #[test]
    #[should_panic(expected = "finite bounds")]
    fn an_infinite_uniform_range_is_rejected() {
        Init::Uniform {
            low: f64::NEG_INFINITY,
            high: 1.0,
        }
        .bounds(1, 1);
    }

    /// Both bounds are finite, but `high - low` overflows, which `Uniform::new` rejects.
    #[test]
    #[should_panic(expected = "finite width")]
    fn a_uniform_range_whose_width_overflows_is_rejected() {
        Init::Uniform {
            low: -f64::MAX,
            high: f64::MAX,
        }
        .bounds(1, 1);
    }
}

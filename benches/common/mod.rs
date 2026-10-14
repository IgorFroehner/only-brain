//! Shared by every benchmark target.

use std::time::Duration;

use criterion::Criterion;
use rand::SeedableRng;
use rand::rngs::StdRng;

use only_brain::NeuralNetwork;

/// A deterministic network, so every run measures the same numbers.
pub fn network<const IN: usize, const OUT: usize>(hidden: &[usize]) -> NeuralNetwork<IN, OUT> {
    let mut rng = StdRng::seed_from_u64(42);
    NeuralNetwork::<IN, OUT>::new_with_rng(hidden, &mut rng)
}

/// Criterion's defaults, with shorter warm-up and measurement times so the whole suite
/// runs in a few minutes. Even the slowest benchmark collects its samples in that time;
/// raise `--measurement-time` on the command line when a change is too close to call.
pub fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

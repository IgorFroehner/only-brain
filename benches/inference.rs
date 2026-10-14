//! Benchmarks for the forward passes and the functions applied to their outputs.
//!
//! `feed_forward` runs once per fitness evaluation in a genetic algorithm, so it is
//! measured across the small shapes such networks actually have, plus a deep narrow
//! shape for the per-layer overhead and two larger shapes to show how the costs scale.
//! `feed_forward_batch` is measured against a loop of `feed_forward` calls over the
//! same inputs, which is what it replaces, on three shapes, and across batch sizes.
//! `feed_forward_by_activation` runs one shape with each activation function.
//! `feed_forward_trace` is the forward pass of backpropagation.
//!
//! Run with `cargo bench --bench inference`, or `cargo bench --bench inference --
//! '^feed_forward/'` for the ids matching a regex.

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group,
    criterion_main,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

use only_brain::{ActivationFunction, Perceptron, argmax, softmax};

mod common;
use common::network;

/// Inputs spread over `[-0.5, 0.5)`, so neurons do not all see the same value.
fn inputs<const N: usize>() -> [f64; N] {
    std::array::from_fn(|i| i as f64 / N as f64 - 0.5)
}

fn bench_feed_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward");

    // The kind of network a steering or game-playing genome drives.
    let tiny = network::<4, 2>(&[6]);
    group.bench_function(BenchmarkId::from_parameter("4-6-2"), |b| {
        let inputs = [0.1, -0.2, 0.3, -0.4];
        b.iter(|| tiny.feed_forward(black_box(&inputs)))
    });

    let small = network::<16, 4>(&[16, 16]);
    group.bench_function(BenchmarkId::from_parameter("16-16-16-4"), |b| {
        let inputs = [0.1; 16];
        b.iter(|| small.feed_forward(black_box(&inputs)))
    });

    // Many layers of few neurons: the per-layer overhead (the scratch swap, the
    // activation dispatch) is a large share of the pass.
    let deep = network::<8, 2>(&[8, 8, 8, 8, 8]);
    group.bench_function(BenchmarkId::from_parameter("8-8-8-8-8-8-2"), |b| {
        let inputs = inputs::<8>();
        b.iter(|| deep.feed_forward(black_box(&inputs)))
    });

    // Wider than the stack scratch, so it takes the heap path.
    let medium = network::<64, 10>(&[128, 64]);
    group.bench_function(BenchmarkId::from_parameter("64-128-64-10"), |b| {
        let inputs = [0.1; 64];
        b.iter(|| medium.feed_forward(black_box(&inputs)))
    });

    // An MNIST-sized classifier: the matrix-vector product dominates.
    let wide = network::<784, 10>(&[128]);
    group.bench_function(BenchmarkId::from_parameter("784-128-10"), |b| {
        let inputs = inputs::<784>();
        b.iter(|| wide.feed_forward(black_box(&inputs)))
    });

    group.finish();
}

/// The same small network with every layer set to each activation function in turn.
/// The `feed_forward` group only runs the default sigmoid, whose cost is mostly `exp`;
/// this one shows what the activation loop costs for the cheaper functions.
fn bench_feed_forward_by_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward_by_activation");
    let inputs = [0.1; 16];

    for activation in ActivationFunction::ALL {
        let mut small = network::<16, 4>(&[16, 16]);
        small.set_activation_function(activation);
        group.bench_function(
            BenchmarkId::new(format!("{activation:?}"), "16-16-16-4"),
            |b| b.iter(|| small.feed_forward(black_box(&inputs))),
        );
    }

    group.finish();
}

/// One network scoring a whole dataset: a loop of single passes against one batch.
fn bench_feed_forward_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward_batch");
    // Benches between about 0.6 and 1.2 ms per iteration take too long for linear
    // sampling to collect its samples in the measurement time, and too little for
    // `Auto` to switch to flat sampling, so they ask for it: the loop over
    // 64-128-64-10 and the batch over 784-128-10.
    use SamplingMode::{Auto, Flat};
    loop_and_batch::<16, 4>(&mut group, &[16, 16], "16-16-16-4", [Auto, Auto]);
    loop_and_batch::<64, 10>(&mut group, &[128, 64], "64-128-64-10", [Flat, Auto]);
    loop_and_batch::<784, 10>(&mut group, &[128], "784-128-10", [Auto, Flat]);
    group.finish();
}

fn loop_and_batch<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    shape: &str,
    [loop_sampling, batch_sampling]: [SamplingMode; 2],
) {
    const SAMPLES: usize = 256;

    let nn = network::<IN, OUT>(hidden);
    let inputs: Vec<[f64; IN]> = (0..SAMPLES)
        .map(|i| [i as f64 / SAMPLES as f64; IN])
        .collect();

    group.sampling_mode(loop_sampling);
    group.bench_function(
        BenchmarkId::new("loop", format!("{shape}/{SAMPLES}")),
        |b| {
            b.iter(|| {
                black_box(&inputs)
                    .iter()
                    .map(|inputs| nn.feed_forward(inputs))
                    .collect::<Vec<_>>()
            })
        },
    );

    group.sampling_mode(batch_sampling);
    group.bench_function(
        BenchmarkId::new("batch", format!("{shape}/{SAMPLES}")),
        |b| b.iter(|| nn.feed_forward_batch(black_box(&inputs))),
    );
}

/// `feed_forward_batch` from a single input up to a large dataset. Throughput is per
/// input, so the report shows where the batch's setup stops mattering.
fn bench_feed_forward_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward_batch_size");
    let small = network::<16, 4>(&[16, 16]);

    // 256 is in `feed_forward_batch`.
    for samples in [1, 16, 64, 4096] {
        let inputs: Vec<[f64; 16]> = (0..samples)
            .map(|i| [i as f64 / samples as f64; 16])
            .collect();
        group.throughput(Throughput::Elements(samples as u64));
        group.bench_function(BenchmarkId::new("16-16-16-4", samples), |b| {
            b.iter(|| small.feed_forward_batch(black_box(&inputs)))
        });
    }

    group.finish();
}

/// The forward pass of backpropagation, which allocates every layer's sums and
/// outputs. 4-8-3 is the network of `examples/iris_backprop.rs`.
fn bench_feed_forward_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("feed_forward_trace");

    let iris = network::<4, 3>(&[8]);
    group.bench_function(BenchmarkId::from_parameter("4-8-3"), |b| {
        let inputs = inputs::<4>();
        b.iter(|| iris.feed_forward_trace(black_box(&inputs)))
    });

    let small = network::<16, 4>(&[16, 16]);
    group.bench_function(BenchmarkId::from_parameter("16-16-16-4"), |b| {
        let inputs = inputs::<16>();
        b.iter(|| small.feed_forward_trace(black_box(&inputs)))
    });

    let medium = network::<64, 10>(&[128, 64]);
    group.bench_function(BenchmarkId::from_parameter("64-128-64-10"), |b| {
        let inputs = inputs::<64>();
        b.iter(|| medium.feed_forward_trace(black_box(&inputs)))
    });

    group.finish();
}

/// `apply` and `derivative` on their own, called one value at a time as user code
/// calls them: backpropagation takes one derivative per neuron.
fn bench_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation");
    const VALUES: usize = 1024;

    // Both sides of zero, and far enough out to reach the saturated ends.
    let values: Vec<f64> = (0..VALUES)
        .map(|i| 8.0 * i as f64 / VALUES as f64 - 4.0)
        .collect();
    let mut results = vec![0.0; VALUES];
    group.throughput(Throughput::Elements(VALUES as u64));

    for activation in ActivationFunction::ALL {
        group.bench_function(BenchmarkId::new("apply", format!("{activation:?}")), |b| {
            b.iter(|| {
                let activation = black_box(activation);
                for (result, &x) in results.iter_mut().zip(black_box(&values)) {
                    *result = activation.apply(x);
                }
                black_box(&results);
            })
        });
        group.bench_function(
            BenchmarkId::new("derivative", format!("{activation:?}")),
            |b| {
                b.iter(|| {
                    let activation = black_box(activation);
                    for (result, &x) in results.iter_mut().zip(black_box(&values)) {
                        *result = activation.derivative(x);
                    }
                    black_box(&results);
                })
            },
        );
    }

    group.finish();
}

/// What turns a classifier's outputs into an answer, once per evaluated input.
fn bench_output_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("output_helpers");
    let logits = inputs::<10>().map(|x| 4.0 * x);

    group.bench_function("softmax/10", |b| b.iter(|| softmax(black_box(&logits))));
    group.bench_function("argmax/10", |b| b.iter(|| argmax(black_box(&logits))));

    group.finish();
}

fn bench_perceptron(c: &mut Criterion) {
    let mut group = c.benchmark_group("perceptron");

    let mut rng = StdRng::seed_from_u64(42);
    let perceptron = Perceptron::<16>::new_with_rng(ActivationFunction::Sigmoid, &mut rng);
    group.bench_function("feed_forward/16", |b| {
        let inputs = inputs::<16>();
        b.iter(|| perceptron.feed_forward(black_box(&inputs)))
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = common::config();
    targets = bench_feed_forward,
        bench_feed_forward_by_activation,
        bench_feed_forward_batch,
        bench_feed_forward_batch_size,
        bench_feed_forward_trace,
        bench_activation,
        bench_output_helpers,
        bench_perceptron
);
criterion_main!(benches);

//! Benchmarks for the operations a training loop leans on.
//!
//! `feed_forward` and the flat parameter view run once per fitness evaluation in a
//! genetic algorithm, so they are measured across the small shapes such networks
//! actually have, plus one larger shape to show how the costs scale.
//! `feed_forward_batch` is measured against a loop of `feed_forward` calls over the
//! same inputs, which is what it replaces. `feed_forward_by_activation` runs one
//! shape with each activation function.
//!
//! Run with `cargo bench`, or `cargo bench -- feed_forward` for one group.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

use only_brain::{ActivationFunction, NeuralNetwork, read_model, write_model};

/// A deterministic network, so every run measures the same numbers.
fn network<const IN: usize, const OUT: usize>(hidden: &[usize]) -> NeuralNetwork<IN, OUT> {
    let mut rng = StdRng::seed_from_u64(42);
    NeuralNetwork::<IN, OUT>::new_with_rng(hidden, &mut rng)
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

    let medium = network::<64, 10>(&[128, 64]);
    group.bench_function(BenchmarkId::from_parameter("64-128-64-10"), |b| {
        let inputs = [0.1; 64];
        b.iter(|| medium.feed_forward(black_box(&inputs)))
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
    const SAMPLES: usize = 256;

    let small = network::<16, 4>(&[16, 16]);
    let small_inputs: Vec<[f64; 16]> = (0..SAMPLES)
        .map(|i| [i as f64 / SAMPLES as f64; 16])
        .collect();
    group.bench_function(
        BenchmarkId::new("loop", format!("16-16-16-4/{SAMPLES}")),
        |b| {
            b.iter(|| {
                black_box(&small_inputs)
                    .iter()
                    .map(|inputs| small.feed_forward(inputs))
                    .collect::<Vec<_>>()
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("batch", format!("16-16-16-4/{SAMPLES}")),
        |b| b.iter(|| small.feed_forward_batch(black_box(&small_inputs))),
    );

    let medium = network::<64, 10>(&[128, 64]);
    let medium_inputs: Vec<[f64; 64]> = (0..SAMPLES)
        .map(|i| [i as f64 / SAMPLES as f64; 64])
        .collect();
    group.bench_function(
        BenchmarkId::new("loop", format!("64-128-64-10/{SAMPLES}")),
        |b| {
            b.iter(|| {
                black_box(&medium_inputs)
                    .iter()
                    .map(|inputs| medium.feed_forward(inputs))
                    .collect::<Vec<_>>()
            })
        },
    );
    group.bench_function(
        BenchmarkId::new("batch", format!("64-128-64-10/{SAMPLES}")),
        |b| b.iter(|| medium.feed_forward_batch(black_box(&medium_inputs))),
    );

    group.finish();
}

/// The genetic algorithm inner loop: genome in, network out, score it.
fn bench_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameters");

    let small = network::<16, 4>(&[16, 16]);
    let genome = small.parameters();

    group.bench_function("parameters/16-16-16-4", |b| {
        b.iter(|| black_box(&small).parameters())
    });

    group.bench_function("set_parameters/16-16-16-4", |b| {
        let mut nn = small.clone();
        b.iter(|| nn.set_parameters(black_box(&genome)))
    });

    group.bench_function("from_parameters/16-16-16-4", |b| {
        b.iter(|| NeuralNetwork::<16, 4>::from_parameters(&[16, 16], black_box(&genome)))
    });

    // Mutating a network: out through the flat view and back, against in place. The
    // mutation is a cheap negation so the copies are what is measured.
    group.bench_function("round_trip_mutation/16-16-16-4", |b| {
        let mut nn = small.clone();
        b.iter(|| {
            let mut genome = black_box(&nn).parameters();
            genome
                .iter_mut()
                .for_each(|parameter| *parameter = -*parameter);
            nn.set_parameters(&genome);
        })
    });

    group.bench_function("map_parameters/16-16-16-4", |b| {
        let mut nn = small.clone();
        b.iter(|| black_box(&mut nn).map_parameters(|parameter| -parameter))
    });

    group.finish();
}

fn bench_model_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_io");

    let medium = network::<64, 10>(&[128, 64]);
    let mut bytes = Vec::new();
    write_model(&medium, &mut bytes).expect("writing to a Vec cannot fail");

    group.bench_function("write/64-128-64-10", |b| {
        let mut buffer = Vec::with_capacity(bytes.len());
        b.iter(|| {
            buffer.clear();
            write_model(black_box(&medium), &mut buffer)
        })
    });

    group.bench_function("read/64-128-64-10", |b| {
        b.iter(|| read_model::<64, 10>(black_box(bytes.as_slice())))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_feed_forward,
    bench_feed_forward_by_activation,
    bench_feed_forward_batch,
    bench_parameters,
    bench_model_io
);
criterion_main!(benches);

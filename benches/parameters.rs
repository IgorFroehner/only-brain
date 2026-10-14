//! Benchmarks for building networks and for both views of their parameters.
//!
//! The flat view is the genetic algorithm inner loop: genome in, network out, score
//! it, mutate. The layer view is what backpropagation reads and writes once per
//! training step. Both are measured on the small shape a genome usually has and on a
//! larger one, to show how the costs scale with the parameter count.
//!
//! Run with `cargo bench --bench parameters`.

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

use only_brain::NeuralNetwork;

mod common;
use common::network;

fn bench_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameters");
    flat_view::<16, 4>(&mut group, &[16, 16], "16-16-16-4");
    flat_view::<64, 10>(&mut group, &[128, 64], "64-128-64-10");
    group.finish();
}

fn flat_view<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    shape: &str,
) {
    let nn = network::<IN, OUT>(hidden);
    let genome = nn.parameters();
    group.throughput(Throughput::Elements(genome.len() as u64));

    group.bench_function(format!("parameters/{shape}"), |b| {
        b.iter(|| black_box(&nn).parameters())
    });

    group.bench_function(format!("set_parameters/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| nn.set_parameters(black_box(&genome)))
    });

    group.bench_function(format!("from_parameters/{shape}"), |b| {
        b.iter(|| NeuralNetwork::<IN, OUT>::from_parameters(hidden, black_box(&genome)))
    });

    // Mutating a network: out through the flat view and back, against in place. The
    // mutation is a cheap negation so the copies are what is measured.
    group.bench_function(format!("round_trip_mutation/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| {
            let mut genome = black_box(&nn).parameters();
            genome
                .iter_mut()
                .for_each(|parameter| *parameter = -*parameter);
            nn.set_parameters(&genome);
        })
    });

    group.bench_function(format!("map_parameters/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| black_box(&mut nn).map_parameters(|parameter| -parameter))
    });

    group.bench_function(format!("for_each_parameter_mut/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| black_box(&mut nn).for_each_parameter_mut(|parameter| *parameter = -*parameter))
    });
}

/// Making networks: a population's first generation, and the copies a genetic
/// algorithm makes of its survivors.
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    construction::<16, 4>(&mut group, &[16, 16], "16-16-16-4");
    construction::<64, 10>(&mut group, &[128, 64], "64-128-64-10");
    group.finish();
}

fn construction<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    shape: &str,
) {
    group.bench_function(format!("new_with_rng/{shape}"), |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| NeuralNetwork::<IN, OUT>::new_with_rng(black_box(hidden), &mut rng))
    });

    let nn = network::<IN, OUT>(hidden);
    group.bench_function(format!("clone/{shape}"), |b| {
        b.iter(|| black_box(&nn).clone())
    });
}

/// Layer 1's weights and biases, read and written the ways backpropagation does: as a
/// whole layer at once, and one weight at a time.
fn bench_layer_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_view");
    layer_view::<16, 4>(&mut group, &[16, 16], "16-16-16-4");
    layer_view::<64, 10>(&mut group, &[128, 64], "64-128-64-10");
    group.finish();
}

fn layer_view<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    shape: &str,
) {
    let nn = network::<IN, OUT>(hidden);
    let neurons = nn.layer_size(1);
    let weights = nn.layer_weights(1);
    let biases = nn.layer_biases(1).to_vec();
    group.throughput(Throughput::Elements((neurons * IN) as u64));

    group.bench_function(format!("layer_weights/{shape}"), |b| {
        b.iter(|| black_box(&nn).layer_weights(1))
    });

    group.bench_function(format!("set_layer_weights/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| nn.set_layer_weights(1, black_box(&weights)))
    });

    group.bench_function(format!("weight_each/{shape}"), |b| {
        b.iter(|| {
            let nn = black_box(&nn);
            let mut sum = 0.0;
            for neuron in 0..neurons {
                for input in 0..IN {
                    sum += nn.weight(1, neuron, input);
                }
            }
            sum
        })
    });

    group.bench_function(format!("set_weight_each/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| {
            for (neuron, row) in black_box(&weights).iter().enumerate() {
                for (input, &weight) in row.iter().enumerate() {
                    nn.set_weight(1, neuron, input, weight);
                }
            }
        })
    });

    group.throughput(Throughput::Elements(neurons as u64));
    group.bench_function(format!("set_layer_biases/{shape}"), |b| {
        let mut nn = nn.clone();
        b.iter(|| nn.set_layer_biases(1, black_box(&biases)))
    });
}

criterion_group!(
    name = benches;
    config = common::config();
    targets = bench_parameters, bench_construction, bench_layer_view
);
criterion_main!(benches);

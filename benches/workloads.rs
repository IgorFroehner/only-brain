//! End-to-end benchmarks: the two training loops the crate is built for, written
//! against the public API as a user would write them. They catch changes that speed up
//! one operation and slow down the loop around it.
//!
//! `ga_generation` is one generation of a genetic algorithm: copy each parent into a
//! child, mutate the child, then score it on a dataset. `genome` keeps the population
//! as `Vec<f64>` genomes and builds each network with `from_parameters`, as
//! `examples/xor_evolution.rs` does. `in_place` keeps networks, copies with
//! `clone_from` into a reused population and mutates with `for_each_parameter_mut`;
//! `in_place_batch` scores with `feed_forward_batch` instead. All three make the same
//! random draws. Choosing the parents is left out: child `i` copies parent `i`, since
//! how a GA picks parents never touches the library.
//!
//! `backprop_epoch` is one epoch of stochastic gradient descent with the training step
//! of `examples/iris_backprop.rs`: on the Iris data, standardized as there but in file
//! order, and on a larger synthetic set.
//!
//! Every iteration starts from the same parents or network and the same seed, so
//! every run does the same work.
//!
//! Run with `cargo bench --bench workloads`.

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::hint::black_box;

use only_brain::{ActivationFunction, NeuralNetwork, softmax};

mod common;
use common::network;

const POPULATION: usize = 100;
const MUTATION_RATE: f64 = 0.2;
const MUTATION_STRENGTH: f64 = 0.4;

/// Inputs and the outputs a network should produce for them, drawn from a seed.
struct Cases<const IN: usize, const OUT: usize> {
    inputs: Vec<[f64; IN]>,
    targets: Vec<[f64; OUT]>,
}

impl<const IN: usize, const OUT: usize> Cases<IN, OUT> {
    fn random(count: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1);
        Cases {
            inputs: (0..count)
                .map(|_| std::array::from_fn(|_| rng.random_range(-1.0..1.0)))
                .collect(),
            targets: (0..count)
                .map(|_| std::array::from_fn(|_| rng.random_range(0.0..1.0)))
                .collect(),
        }
    }

    /// Lower is better: the summed squared error over every case.
    fn error(&self, nn: &NeuralNetwork<IN, OUT>) -> f64 {
        let outputs = self.inputs.iter().map(|inputs| nn.feed_forward(inputs));
        Self::squared_error(outputs, &self.targets)
    }

    /// The same, with all the cases in one `feed_forward_batch`.
    fn error_batch(&self, nn: &NeuralNetwork<IN, OUT>) -> f64 {
        Self::squared_error(nn.feed_forward_batch(&self.inputs), &self.targets)
    }

    fn squared_error(outputs: impl IntoIterator<Item = [f64; OUT]>, targets: &[[f64; OUT]]) -> f64 {
        outputs
            .into_iter()
            .zip(targets)
            .flat_map(|(outputs, targets)| outputs.into_iter().zip(targets))
            .map(|(output, target)| (output - target).powi(2))
            .sum()
    }
}

/// Gaussian mutation of one gene, with probability `MUTATION_RATE`.
fn mutate(gene: &mut f64, noise: &Normal<f64>, rng: &mut impl Rng) {
    if rng.random_bool(MUTATION_RATE) {
        *gene += noise.sample(rng);
    }
}

fn bench_ga_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_generation");
    // A steering genome scored on a handful of situations, and a larger one scored on
    // a small dataset.
    ga_generation::<4, 2>(&mut group, &[6], 16, "4-6-2");
    ga_generation::<16, 4>(&mut group, &[16, 16], 64, "16-16-16-4");
    group.finish();
}

fn ga_generation<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    cases: usize,
    shape: &str,
) {
    let cases = Cases::<IN, OUT>::random(cases);
    let noise = Normal::new(0.0, MUTATION_STRENGTH).expect("the deviation is positive");

    let mut rng = StdRng::seed_from_u64(42);
    let networks: Vec<NeuralNetwork<IN, OUT>> = (0..POPULATION)
        .map(|_| NeuralNetwork::new_with_rng(hidden, &mut rng))
        .collect();
    let genomes: Vec<Vec<f64>> = networks.iter().map(NeuralNetwork::parameters).collect();

    group.bench_function(BenchmarkId::new("genome", shape), |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(7);
            genomes
                .iter()
                .map(|parent| {
                    let mut child = parent.clone();
                    child
                        .iter_mut()
                        .for_each(|gene| mutate(gene, &noise, &mut rng));
                    let nn = NeuralNetwork::<IN, OUT>::from_parameters(hidden, &child);
                    let error = cases.error(&nn);
                    (child, error)
                })
                .collect::<Vec<_>>()
        })
    });

    let mut children = networks.clone();
    group.bench_function(BenchmarkId::new("in_place", shape), |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(7);
            children
                .iter_mut()
                .zip(&networks)
                .map(|(child, parent)| {
                    child.clone_from(parent);
                    child.for_each_parameter_mut(|gene| mutate(gene, &noise, &mut rng));
                    cases.error(child)
                })
                .collect::<Vec<_>>()
        })
    });

    group.bench_function(BenchmarkId::new("in_place_batch", shape), |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(7);
            children
                .iter_mut()
                .zip(&networks)
                .map(|(child, parent)| {
                    child.clone_from(parent);
                    child.for_each_parameter_mut(|gene| mutate(gene, &noise, &mut rng));
                    cases.error_batch(child)
                })
                .collect::<Vec<_>>()
        })
    });
}

/// One labelled example for classification.
struct Sample<const IN: usize> {
    features: [f64; IN],
    class: usize,
}

/// The 150 flowers of `examples/datasets/iris.csv`, in file order, each feature
/// rescaled to zero mean and unit variance as the example does, so the tanh layer
/// works in its useful range rather than saturated.
fn iris() -> Vec<Sample<4>> {
    const SPECIES: [&str; 3] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

    // Columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    let mut samples: Vec<Sample<4>> = include_str!("../examples/datasets/iris.csv")
        .lines()
        .skip(1)
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            Sample {
                features: std::array::from_fn(|i| {
                    fields[i + 1].parse().expect("the measurements are numbers")
                }),
                class: SPECIES
                    .iter()
                    .position(|&name| name == fields[5])
                    .expect("every species is known"),
            }
        })
        .collect();

    for feature in 0..4 {
        let n = samples.len() as f64;
        let mean = samples.iter().map(|s| s.features[feature]).sum::<f64>() / n;
        let variance = samples
            .iter()
            .map(|s| (s.features[feature] - mean).powi(2))
            .sum::<f64>()
            / n;
        for sample in &mut samples {
            sample.features[feature] = (sample.features[feature] - mean) / variance.sqrt();
        }
    }

    samples
}

/// One step of stochastic gradient descent on one sample, minimising the
/// cross-entropy of a softmax over the outputs. The same step as
/// `examples/iris_backprop.rs`, for any shape. Returns the loss before the step.
fn train_step<const IN: usize, const OUT: usize>(
    nn: &mut NeuralNetwork<IN, OUT>,
    sample: &Sample<IN>,
    learning_rate: f64,
) -> f64 {
    let trace = nn.feed_forward_trace(&sample.features);
    let logits: &[f64; OUT] = trace.output().try_into().expect("the output is OUT wide");
    let probabilities = softmax(logits);
    let loss = -probabilities[sample.class].ln();

    let mut output_gradient = probabilities.to_vec();
    output_gradient[sample.class] -= 1.0;

    for layer in (1..nn.num_layers()).rev() {
        let activation = nn.layer_activation(layer);
        let inputs = trace.outputs(layer - 1);

        let delta: Vec<f64> = output_gradient
            .iter()
            .zip(trace.pre_activations(layer))
            .map(|(gradient, &sum)| gradient * activation.derivative(sum))
            .collect();

        let mut weights = nn.layer_weights(layer);

        if layer > 1 {
            output_gradient = (0..inputs.len())
                .map(|input| {
                    delta
                        .iter()
                        .zip(&weights)
                        .map(|(d, row)| d * row[input])
                        .sum()
                })
                .collect();
        }

        for (row, d) in weights.iter_mut().zip(&delta) {
            for (weight, x) in row.iter_mut().zip(inputs) {
                *weight -= learning_rate * d * x;
            }
        }
        let biases: Vec<f64> = nn
            .layer_biases(layer)
            .iter()
            .zip(&delta)
            .map(|(bias, d)| bias - learning_rate * d)
            .collect();

        nn.set_layer_weights(layer, &weights);
        nn.set_layer_biases(layer, &biases);
    }

    loss
}

fn bench_backprop_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("backprop_epoch");

    let iris = iris();
    let mut classifier = network::<4, 3>(&[8]);
    classifier.set_activation_function(ActivationFunction::Tanh);
    classifier.set_output_activation(ActivationFunction::Identity);
    backprop_epoch(&mut group, &classifier, &iris, "iris/4-8-3");

    let mut rng = StdRng::seed_from_u64(1);
    let synthetic: Vec<Sample<64>> = (0..128)
        .map(|i| Sample {
            features: std::array::from_fn(|_| rng.random_range(-1.0..1.0)),
            class: i % 10,
        })
        .collect();
    let mut classifier = network::<64, 10>(&[128, 64]);
    classifier.set_activation_function(ActivationFunction::ReLU);
    classifier.set_output_activation(ActivationFunction::Identity);
    backprop_epoch(
        &mut group,
        &classifier,
        &synthetic,
        "synthetic/64-128-64-10",
    );

    group.finish();
}

fn backprop_epoch<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    classifier: &NeuralNetwork<IN, OUT>,
    samples: &[Sample<IN>],
    name: &str,
) {
    const LEARNING_RATE: f64 = 0.02;

    group.bench_function(format!("{name}/{}", samples.len()), |b| {
        b.iter_batched(
            || classifier.clone(),
            |mut nn| {
                let loss: f64 = black_box(samples)
                    .iter()
                    .map(|sample| train_step(&mut nn, sample, LEARNING_RATE))
                    .sum();
                (nn, loss)
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    name = benches;
    config = common::config();
    targets = bench_ga_generation, bench_backprop_epoch
);
criterion_main!(benches);

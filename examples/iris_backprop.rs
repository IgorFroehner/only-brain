//! Trains a 4 -> 8 -> 3 network to classify the three Iris species with
//! backpropagation and plain stochastic gradient descent, written against the public
//! API only.
//!
//! The library has no training code: `feed_forward_trace` gives every layer's weighted
//! sums and outputs, `ActivationFunction::derivative` gives each activation's slope,
//! and the layer-by-layer getters and setters read and write the weights. Everything
//! else, the loss, the gradients and the update, lives in this file.
//!
//! The output layer uses `Identity`, so its outputs are logits; the example turns
//! them into probabilities with a softmax and minimises the cross-entropy.
//!
//! Run with `cargo run --example iris_backprop`.

use std::error::Error;
use std::fs;

use only_brain::{argmax, softmax, ActivationFunction, NeuralNetwork};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

type Classifier = NeuralNetwork<4, 3>;

const SPECIES: [&str; 3] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
const HIDDEN: &[usize] = &[8];

const SEED: u64 = 7;
const TEST_SAMPLES: usize = 30;
const EPOCHS: usize = 100;
const LEARNING_RATE: f64 = 0.02;
/// The held-out accuracy the trained network must reach, so CI catches regressions.
const REQUIRED_TEST_ACCURACY: f64 = 0.9;

/// One flower: its four measurements in centimetres and the index of its species in
/// `SPECIES`.
#[derive(Clone)]
struct Sample {
    features: [f64; 4],
    species: usize,
}

/// Reads every row of the Iris CSV.
fn load_samples(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut samples = Vec::new();

    // Columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    for line in fs::read_to_string(path)?.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let species = SPECIES
            .iter()
            .position(|&name| name == fields[5])
            .ok_or_else(|| format!("unknown species {:?}", fields[5]))?;

        samples.push(Sample {
            features: [fields[1].parse()?, fields[2].parse()?, fields[3].parse()?, fields[4].parse()?],
            species,
        });
    }

    Ok(samples)
}

/// Rescales every feature to zero mean and unit variance, using the training set's
/// statistics for both sets so the test set stays unseen.
fn standardize(train: &mut [Sample], test: &mut [Sample]) {
    for feature in 0..4 {
        let n = train.len() as f64;
        let mean = train.iter().map(|s| s.features[feature]).sum::<f64>() / n;
        let variance = train.iter().map(|s| (s.features[feature] - mean).powi(2)).sum::<f64>() / n;
        let deviation = variance.sqrt();

        for sample in train.iter_mut().chain(test.iter_mut()) {
            sample.features[feature] = (sample.features[feature] - mean) / deviation;
        }
    }
}

/// One step of stochastic gradient descent on one sample. Returns the sample's
/// cross-entropy loss before the step.
fn train_step(nn: &mut Classifier, sample: &Sample, learning_rate: f64) -> f64 {
    let trace = nn.feed_forward_trace(&sample.features);
    let logits: &[f64; 3] = trace.output().try_into().expect("there are three classes");
    let probabilities = softmax(logits);
    let loss = -probabilities[sample.species].ln();

    // The gradient of softmax + cross-entropy with respect to the logits, which are
    // the output layer's outputs: the probabilities minus the one-hot target.
    let mut output_gradient = probabilities.to_vec();
    output_gradient[sample.species] -= 1.0;

    // Walk back from the output layer to layer 1. `output_gradient` holds the loss's
    // gradient with respect to the current layer's outputs.
    for layer in (1..nn.num_layers()).rev() {
        let activation = nn.layer_activation(layer);
        let inputs = trace.outputs(layer - 1);

        // The gradient with respect to each neuron's weighted sum.
        let delta: Vec<f64> = output_gradient
            .iter()
            .zip(trace.pre_activations(layer))
            .map(|(gradient, &sum)| gradient * activation.derivative(sum))
            .collect();

        let mut weights = nn.layer_weights(layer);

        // Pass the gradient on to the previous layer's outputs, through the weights
        // as they were in the forward pass. Layer 1's previous layer is the inputs,
        // which have nothing to train.
        if layer > 1 {
            output_gradient = (0..inputs.len())
                .map(|input| delta.iter().zip(&weights).map(|(d, row)| d * row[input]).sum())
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

fn accuracy(nn: &Classifier, samples: &[Sample]) -> f64 {
    let inputs: Vec<[f64; 4]> = samples.iter().map(|s| s.features).collect();
    let correct = nn
        .feed_forward_batch(&inputs)
        .iter()
        .zip(samples)
        .filter(|(logits, sample)| argmax(logits) == sample.species)
        .count();

    correct as f64 / samples.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut samples = load_samples("examples/datasets/iris.csv")?;
    samples.shuffle(&mut rng);
    let mut test = samples.split_off(samples.len() - TEST_SAMPLES);
    let mut train = samples;
    standardize(&mut train, &mut test);
    println!("Loaded {} training and {} test samples", train.len(), test.len());

    let mut nn = Classifier::new_with_rng(HIDDEN, &mut rng);
    nn.set_activation_function(ActivationFunction::Tanh);
    nn.set_output_activation(ActivationFunction::Identity);
    println!("Training a 4 -> {HIDDEN:?} -> 3 network for {EPOCHS} epochs\n");

    for epoch in 1..=EPOCHS {
        train.shuffle(&mut rng);
        let loss: f64 = train.iter().map(|sample| train_step(&mut nn, sample, LEARNING_RATE)).sum();

        if epoch == 1 || epoch % 10 == 0 {
            println!(
                "epoch {epoch:3}: mean loss {:.4}, train accuracy {:5.1}%, test accuracy {:5.1}%",
                loss / train.len() as f64,
                100.0 * accuracy(&nn, &train),
                100.0 * accuracy(&nn, &test),
            );
        }
    }

    let test_accuracy = accuracy(&nn, &test);
    println!("\nTest accuracy: {:.1}%", 100.0 * test_accuracy);
    assert!(
        test_accuracy >= REQUIRED_TEST_ACCURACY,
        "test accuracy {:.1}% is below the required {:.1}%",
        100.0 * test_accuracy,
        100.0 * REQUIRED_TEST_ACCURACY,
    );

    Ok(())
}

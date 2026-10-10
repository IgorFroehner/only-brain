//! Trains a perceptron to tell Iris setosa from Iris versicolor by their sepals, with
//! the classic perceptron learning rule.
//!
//! Run with `cargo run --example perceptron_iris`.

use std::error::Error;
use std::fs;

use only_brain::{bvector, ActivationFunction, Perceptron};

/// One flower: sepal length and width in centimetres, and 1.0 for versicolor or 0.0
/// for setosa.
struct Sample {
    sepal_length: f64,
    sepal_width: f64,
    is_versicolor: f64,
}

/// Reads the setosa and versicolor rows of the Iris CSV and drops the other species.
fn load_samples(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let mut samples = Vec::new();

    // Columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
    for line in fs::read_to_string(path)?.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let is_versicolor = match fields[5] {
            "Iris-setosa" => 0.0,
            "Iris-versicolor" => 1.0,
            _ => continue,
        };

        samples.push(Sample {
            sepal_length: fields[1].parse()?,
            sepal_width: fields[2].parse()?,
            is_versicolor,
        });
    }

    Ok(samples)
}

fn accuracy(perceptron: &Perceptron<2>, samples: &[Sample]) -> f64 {
    let correct = samples
        .iter()
        .filter(|s| perceptron.feed_forward(&bvector![s.sepal_length, s.sepal_width]) == s.is_versicolor)
        .count();

    correct as f64 / samples.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = load_samples("examples/datasets/iris.csv")?;
    println!("Loaded {} setosa and versicolor samples", samples.len());

    let learning_rate = 0.1;
    let mut weights = [0.0, 0.0];
    let mut bias = 0.0;
    let mut perceptron = Perceptron::<2>::new(ActivationFunction::BinaryStep);

    const MAX_EPOCHS: usize = 1000;
    for epoch in 1..=MAX_EPOCHS {
        let mut mistakes = 0;

        for sample in &samples {
            let inputs = [sample.sepal_length, sample.sepal_width];
            let error = sample.is_versicolor - perceptron.feed_forward(&bvector![inputs[0], inputs[1]]);
            if error == 0.0 {
                continue;
            }

            // Perceptron rule: move the boundary towards each misclassified sample.
            mistakes += 1;
            for (weight, input) in weights.iter_mut().zip(inputs) {
                *weight += learning_rate * error * input;
            }
            bias += learning_rate * error;
            perceptron.set_weights(bvector![weights[0], weights[1]]);
            perceptron.set_bias(bias);
        }

        if mistakes == 0 {
            println!("Converged after {epoch} epochs");
            break;
        }
        if epoch == MAX_EPOCHS {
            println!("Still {mistakes} mistakes after {MAX_EPOCHS} epochs");
        }
    }

    println!("\n{perceptron}");
    println!("Accuracy: {:.1}%", 100.0 * accuracy(&perceptron, &samples));

    Ok(())
}

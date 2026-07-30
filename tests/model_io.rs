//! Public-API tests for dumping and loading a model to disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use nalgebra::{dmatrix, dvector};
use only_brain::{dump_model, load_model, ActivationFunction, NeuralNetwork};

const EPSILON: f64 = 1e-12;

/// Removes its file when dropped, so a failing assertion cannot leak it.
struct TempModelPath(PathBuf);

impl TempModelPath {
    fn new(name: &str) -> Self {
        // Distinguishes files created by tests running in parallel.
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);

        let path = std::env::temp_dir().join(format!(
            "only-brain-{}-{}-{}.bin",
            name,
            std::process::id(),
            unique
        ));
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempModelPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn sample_network() -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(&vec![2, 3, 2]);

    nn.set_layer_weights(
        1,
        dmatrix![0.1, 0.2;
                 0.3, 0.4;
                 0.5, 0.6],
    );
    nn.set_layer_biases(1, dvector![0.1, 0.2, 0.3]);

    nn.set_layer_weights(
        2,
        dmatrix![0.9, 0.8, 0.7;
                 0.6, 0.5, 0.4],
    );
    nn.set_layer_biases(2, dvector![0.1, 0.2]);

    nn
}

fn assert_all_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < EPSILON, "at index {i}: expected {e}, got {a}");
    }
}

#[test]
fn a_dumped_model_loads_back_with_identical_structure_and_weights() {
    let temp = TempModelPath::new("round-trip");
    let original = sample_network();

    dump_model(&original, temp.path().to_str().unwrap()).expect("dump should succeed");
    let loaded = load_model(temp.path().to_str().unwrap()).expect("load should succeed");

    assert_eq!(loaded.num_layers(), original.num_layers());
    for layer in 0..original.num_layers() {
        assert_eq!(loaded.layer_size(layer), original.layer_size(layer));
    }

    for layer in 1..original.num_layers() {
        for neuron in 0..original.layer_size(layer) {
            for input in 0..original.layer_size(layer - 1) {
                assert_eq!(
                    loaded.get_weight(layer, neuron, input),
                    original.get_weight(layer, neuron, input),
                    "weight mismatch at layer {layer}, neuron {neuron}, input {input}"
                );
            }
        }
    }
}

#[test]
fn a_loaded_model_produces_the_same_output_as_the_original() {
    let temp = TempModelPath::new("output");
    let original = sample_network();
    let input = vec![0.5, 0.2];
    let expected = original.feed_forward(&input);

    dump_model(&original, temp.path().to_str().unwrap()).expect("dump should succeed");
    let loaded = load_model(temp.path().to_str().unwrap()).expect("load should succeed");

    assert_all_close(&loaded.feed_forward(&input), &expected);
}

#[test]
fn the_activation_function_survives_a_round_trip() {
    let temp = TempModelPath::new("activation");
    let mut original = sample_network();
    original.set_activation_function(ActivationFunction::BinaryStep);

    dump_model(&original, temp.path().to_str().unwrap()).expect("dump should succeed");
    let loaded = load_model(temp.path().to_str().unwrap()).expect("load should succeed");

    assert_eq!(loaded.activation_function(), ActivationFunction::BinaryStep);
    assert_all_close(
        &loaded.feed_forward(&vec![0.5, 0.2]),
        &original.feed_forward(&vec![0.5, 0.2]),
    );
}

#[test]
fn loading_a_missing_file_returns_an_error_instead_of_panicking() {
    let missing = std::env::temp_dir().join("only-brain-does-not-exist.bin");
    let _ = fs::remove_file(&missing);

    assert!(load_model(missing.to_str().unwrap()).is_err());
}

#[test]
fn loading_a_file_that_is_not_a_model_returns_an_error() {
    let temp = TempModelPath::new("garbage");
    fs::write(temp.path(), b"this is not a bincode-encoded model").unwrap();

    assert!(load_model(temp.path().to_str().unwrap()).is_err());
}

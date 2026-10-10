//! Public-API tests for dumping and loading a model to disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use nalgebra::{dmatrix, dvector, DMatrix, DVector};
use only_brain::{dump_model, load_model, ActivationFunction, ModelError, NeuralNetwork};

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

fn sample_network() -> NeuralNetwork<2, 2> {
    let mut nn = NeuralNetwork::<2, 2>::new(&[3]);

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

    dump_model(&original, temp.path()).expect("dump should succeed");
    let loaded: NeuralNetwork<2, 2> =
        load_model(temp.path()).expect("load should succeed");

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
    let input = [0.5, 0.2];
    let expected = original.feed_forward(&input);

    dump_model(&original, temp.path()).expect("dump should succeed");
    let loaded: NeuralNetwork<2, 2> =
        load_model(temp.path()).expect("load should succeed");

    assert_all_close(&loaded.feed_forward(&input), &expected);
}

#[test]
fn the_activation_function_survives_a_round_trip() {
    let temp = TempModelPath::new("activation");
    let mut original = sample_network();
    original.set_activation_function(ActivationFunction::BinaryStep);

    dump_model(&original, temp.path()).expect("dump should succeed");
    let loaded: NeuralNetwork<2, 2> =
        load_model(temp.path()).expect("load should succeed");

    assert_eq!(loaded.activation_function(), ActivationFunction::BinaryStep);
    assert_all_close(
        &loaded.feed_forward(&[0.5, 0.2]),
        &original.feed_forward(&[0.5, 0.2]),
    );
}

#[test]
fn loading_with_the_wrong_output_width_reports_a_dimension_mismatch() {
    let temp = TempModelPath::new("wrong-out");
    dump_model(&sample_network(), temp.path()).expect("dump should succeed");

    // The file holds a 2 -> 3 -> 2 network; ask for 2 -> 1.
    let result = load_model::<2, 1>(temp.path());

    match result {
        Err(ModelError::DimensionMismatch {
            end,
            expected,
            found,
        }) => {
            assert_eq!(end, "output");
            assert_eq!(expected, 1);
            assert_eq!(found, 2);
        }
        other => panic!("expected a dimension mismatch, got {other:?}"),
    }
}

#[test]
fn loading_with_the_wrong_input_width_reports_a_dimension_mismatch() {
    let temp = TempModelPath::new("wrong-in");
    dump_model(&sample_network(), temp.path()).expect("dump should succeed");

    let result = load_model::<5, 2>(temp.path());

    match result {
        Err(ModelError::DimensionMismatch {
            end,
            expected,
            found,
        }) => {
            assert_eq!(end, "input");
            assert_eq!(expected, 5);
            assert_eq!(found, 2);
        }
        other => panic!("expected a dimension mismatch, got {other:?}"),
    }
}

/// Mirrors the on-disk layout of a network so tests can write files that the
/// public API would never produce.
#[derive(serde::Serialize)]
struct RawLayer {
    size: usize,
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

#[derive(serde::Serialize)]
struct RawNetwork {
    layers: Vec<RawLayer>,
    activation_function: Option<ActivationFunction>,
}

fn raw_layer(neurons: usize, inputs: usize) -> RawLayer {
    RawLayer {
        size: neurons,
        weights: DMatrix::from_element(neurons, inputs, 0.5),
        bias: DVector::from_element(neurons, 0.1),
    }
}

fn write_raw(temp: &TempModelPath, layers: Vec<RawLayer>) {
    let raw = RawNetwork {
        layers,
        activation_function: None,
    };
    fs::write(temp.path(), bincode::serialize(&raw).unwrap()).unwrap();
}

fn assert_inconsistent_layer(result: Result<NeuralNetwork<2, 2>, ModelError>, expected_layer: usize) {
    match result {
        Err(ModelError::InconsistentLayer { layer, .. }) => assert_eq!(layer, expected_layer),
        other => panic!("expected an inconsistent layer error, got {other:?}"),
    }
}

#[test]
fn the_raw_mirror_matches_the_real_format() {
    let temp = TempModelPath::new("raw-mirror");
    write_raw(&temp, vec![raw_layer(3, 2), raw_layer(2, 3)]);

    let loaded = load_model::<2, 2>(temp.path()).expect("a well-formed raw file should load");

    assert_eq!(loaded.layer_size(1), 3);
}

#[test]
fn loading_rejects_hidden_layers_that_do_not_chain() {
    let temp = TempModelPath::new("broken-chain");
    // 2 -> 3, then a layer expecting 4 inputs.
    write_raw(&temp, vec![raw_layer(3, 2), raw_layer(2, 4)]);

    assert_inconsistent_layer(load_model(temp.path()), 2);
}

#[test]
fn loading_rejects_biases_that_do_not_match_the_weights() {
    let temp = TempModelPath::new("bad-bias");
    let mut first = raw_layer(3, 2);
    first.bias = DVector::from_element(2, 0.1);
    write_raw(&temp, vec![first, raw_layer(2, 3)]);

    assert_inconsistent_layer(load_model(temp.path()), 1);
}

/// The output check used to trust the stored `size` field, so this file loaded
/// as a 2 -> 1 network and then panicked in `feed_forward`.
#[test]
fn loading_rejects_a_size_field_that_disagrees_with_the_weights() {
    let temp = TempModelPath::new("lying-size");
    let mut last = raw_layer(2, 3);
    last.size = 1;
    write_raw(&temp, vec![raw_layer(3, 2), last]);

    assert!(load_model::<2, 1>(temp.path()).is_err());
    assert_inconsistent_layer(load_model(temp.path()), 2);
}

/// Validation lives in `Deserialize` itself, so other serde entry points cannot
/// skip it.
#[test]
fn deserializing_directly_also_validates() {
    let raw = RawNetwork {
        layers: vec![raw_layer(3, 2), raw_layer(2, 4)],
        activation_function: None,
    };
    let bytes = bincode::serialize(&raw).unwrap();

    assert!(bincode::deserialize::<NeuralNetwork<2, 2>>(&bytes).is_err());
}

#[test]
fn loading_a_missing_file_returns_an_error_instead_of_panicking() {
    let missing = std::env::temp_dir().join("only-brain-does-not-exist.bin");
    let _ = fs::remove_file(&missing);

    assert!(load_model::<2, 2>(&missing).is_err());
}

#[test]
fn loading_a_file_that_is_not_a_model_returns_an_error() {
    let temp = TempModelPath::new("garbage");
    fs::write(temp.path(), b"this is not a bincode-encoded model").unwrap();

    assert!(load_model::<2, 2>(temp.path()).is_err());
}

/// Written by 0.1.x (`examples/neural_network.rs`). The format must stay loadable.
#[test]
fn a_model_written_by_an_earlier_version_still_loads() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0_1_model.bin");

    let loaded: NeuralNetwork<2, 2> = load_model(path).expect("0.1.x model should load");

    assert_eq!(loaded.layer_size(1), 3);
    assert_eq!(loaded.activation_function(), ActivationFunction::BinaryStep);
    assert_eq!(loaded.get_weight(1, 0, 0), 0.99);
    assert_eq!(loaded.get_weight(2, 1, 2), 0.4);
}

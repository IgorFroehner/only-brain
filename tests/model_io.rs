//! Public-API tests for dumping and loading a model to disk.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use nalgebra::{DMatrix, DVector};
use only_brain::{
    dump_model, load_model, read_model, write_model, ActivationFunction, ModelError,
    NeuralNetwork, MODEL_FORMAT_VERSION,
};

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

    nn.set_layer_weights(1, &[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
    nn.set_layer_biases(1, &[0.1, 0.2, 0.3]);

    nn.set_layer_weights(2, &[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]);
    nn.set_layer_biases(2, &[0.1, 0.2]);

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
                    loaded.weight(layer, neuron, input),
                    original.weight(layer, neuron, input),
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
fn per_layer_activation_functions_survive_a_round_trip() {
    let temp = TempModelPath::new("activation");
    let mut original = sample_network();
    original.set_activation_function(ActivationFunction::ReLU);
    original.set_output_activation(ActivationFunction::Tanh);

    dump_model(&original, temp.path()).expect("dump should succeed");
    let loaded: NeuralNetwork<2, 2> =
        load_model(temp.path()).expect("load should succeed");

    assert_eq!(loaded.layer_activation(1), ActivationFunction::ReLU);
    assert_eq!(loaded.layer_activation(2), ActivationFunction::Tanh);
    assert_eq!(loaded, original);
}

/// Also pins each function's byte in the file and its name in the serde form, both
/// of which are part of the format and must never change.
#[test]
fn every_activation_function_survives_a_round_trip() {
    let expected = [
        (ActivationFunction::Sigmoid, 0, "Sigmoid"),
        (ActivationFunction::Tanh, 1, "Tanh"),
        (ActivationFunction::ReLU, 2, "ReLU"),
        (ActivationFunction::BinaryStep, 3, "BinaryStep"),
        (ActivationFunction::Identity, 4, "Identity"),
        (ActivationFunction::LeakyReLU, 5, "LeakyReLU"),
        (ActivationFunction::ELU, 6, "ELU"),
        (ActivationFunction::Softplus, 7, "Softplus"),
        (ActivationFunction::SiLU, 8, "SiLU"),
    ];
    assert_eq!(expected.len(), ActivationFunction::ALL.len());

    for (activation, code, name) in expected {
        let mut original = sample_network();
        original.set_activation_function(activation);

        let bytes = to_bytes(&original);
        // magic (8) + version (2) + inputs (4) + layers (4) + first layer's neurons (4)
        assert_eq!(bytes[22], code, "{activation:?}");

        let loaded: NeuralNetwork<2, 2> = read_model(bytes.as_slice()).expect("read should succeed");
        assert_eq!(loaded.layer_activation(1), activation);
        assert_eq!(loaded.output_activation(), activation);
        assert_eq!(loaded, original);

        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains(&format!(r#""activation":"{name}""#)), "{json}");
        let from_json: NeuralNetwork<2, 2> = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, original);
    }
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

/// Mirrors the pre-0.3 on-disk layout of a network, the bincode 1 encoding of nalgebra
/// matrices, so tests can write legacy files, including ones the old API would never
/// have produced.
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
    write_raw_with_activation(temp, layers, None);
}

fn write_raw_with_activation(
    temp: &TempModelPath,
    layers: Vec<RawLayer>,
    activation_function: Option<ActivationFunction>,
) {
    let raw = RawNetwork {
        layers,
        activation_function,
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
fn a_legacy_file_loads() {
    let temp = TempModelPath::new("raw-mirror");
    write_raw(&temp, vec![raw_layer(3, 2), raw_layer(2, 3)]);

    let loaded = load_model::<2, 2>(temp.path()).expect("a well-formed legacy file should load");

    assert_eq!(loaded.layer_size(1), 3);
    assert_eq!(loaded.layer_weights(1), vec![vec![0.5; 2]; 3]);
    assert_eq!(loaded.layer_biases(2), &[0.1, 0.1]);
    // No stored activation meant sigmoid.
    assert_eq!(loaded.layer_activation(1), ActivationFunction::Sigmoid);
    assert_eq!(loaded.output_activation(), ActivationFunction::Sigmoid);
}

/// Legacy files store the matrix column by column; the loader must not transpose it.
#[test]
fn a_legacy_file_keeps_each_weight_in_its_place() {
    let temp = TempModelPath::new("raw-order");
    let mut first = raw_layer(3, 2);
    first.weights = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    write_raw(&temp, vec![first, raw_layer(2, 3)]);

    let loaded = load_model::<2, 2>(temp.path()).expect("legacy file should load");

    assert_eq!(
        loaded.layer_weights(1),
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
    );
}

/// Legacy files had one activation for the whole network; every layer now gets it.
#[test]
fn a_legacy_network_wide_activation_applies_to_every_layer() {
    let temp = TempModelPath::new("raw-activation");
    write_raw_with_activation(
        &temp,
        vec![raw_layer(3, 2), raw_layer(2, 3)],
        Some(ActivationFunction::Tanh),
    );

    let loaded = load_model::<2, 2>(temp.path()).expect("legacy file should load");

    assert_eq!(loaded.layer_activation(1), ActivationFunction::Tanh);
    assert_eq!(loaded.layer_activation(2), ActivationFunction::Tanh);
}

#[test]
fn every_pre_0_3_activation_loads_from_a_legacy_file() {
    for activation in [
        ActivationFunction::Sigmoid,
        ActivationFunction::Tanh,
        ActivationFunction::ReLU,
        ActivationFunction::BinaryStep,
    ] {
        let temp = TempModelPath::new("raw-old-activation");
        write_raw_with_activation(&temp, vec![raw_layer(3, 2), raw_layer(2, 3)], Some(activation));

        let loaded = load_model::<2, 2>(temp.path()).expect("legacy file should load");

        assert_eq!(loaded.output_activation(), activation);
    }
}

/// The legacy format is frozen at the four functions that existed before 0.3, so a
/// later function's index (Identity is 4) there means the file is not a legacy model,
/// rather than growing the old format with the new code table.
#[test]
fn a_legacy_file_with_a_function_added_after_0_3_is_unrecognized() {
    for activation in [
        ActivationFunction::Identity,
        ActivationFunction::LeakyReLU,
        ActivationFunction::SiLU,
    ] {
        let temp = TempModelPath::new("raw-new-activation");
        write_raw_with_activation(&temp, vec![raw_layer(3, 2), raw_layer(2, 3)], Some(activation));

        assert!(
            matches!(load_model::<2, 2>(temp.path()), Err(ModelError::UnrecognizedFormat)),
            "{activation:?}"
        );
    }
}

#[test]
fn a_legacy_file_with_no_layers_reports_an_empty_network() {
    let temp = TempModelPath::new("raw-empty");
    write_raw(&temp, vec![]);

    assert!(matches!(load_model::<2, 2>(temp.path()), Err(ModelError::EmptyNetwork)));
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

fn raw_bytes(layers: Vec<RawLayer>) -> Vec<u8> {
    let raw = RawNetwork {
        layers,
        activation_function: None,
    };
    bincode::serialize(&raw).unwrap()
}

/// Overwrites the `u64` at `offset`, checking first that it holds `old` so a wrong
/// offset fails here rather than corrupting some other field.
fn replace_u64(bytes: &mut [u8], offset: usize, old: u64, new: u64) {
    let field = &mut bytes[offset..offset + 8];
    assert_eq!(u64::from_le_bytes(field.try_into().unwrap()), old, "unexpected value at {offset}");
    field.copy_from_slice(&new.to_le_bytes());
}

// Offsets into a legacy 2 -> 3 -> 2 file: layer count (8), then the first layer's
// size (8), weights length (8), 6 weights (48), rows (8), columns (8), biases
// length (8), 3 biases (24) and biases rows (8).
const LEGACY_FIRST_WEIGHT_ROWS: usize = 72;
const LEGACY_FIRST_BIAS_ROWS: usize = 120;

#[test]
fn a_legacy_matrix_whose_shape_disagrees_with_its_values_is_unrecognized() {
    let mut bytes = raw_bytes(vec![raw_layer(3, 2), raw_layer(2, 3)]);
    replace_u64(&mut bytes, LEGACY_FIRST_WEIGHT_ROWS, 3, 4);

    assert!(matches!(
        read_model::<2, 2>(bytes.as_slice()),
        Err(ModelError::UnrecognizedFormat)
    ));
}

#[test]
fn a_legacy_bias_vector_whose_length_disagrees_with_its_values_is_unrecognized() {
    let mut bytes = raw_bytes(vec![raw_layer(3, 2), raw_layer(2, 3)]);
    replace_u64(&mut bytes, LEGACY_FIRST_BIAS_ROWS, 3, 4);

    assert!(matches!(
        read_model::<2, 2>(bytes.as_slice()),
        Err(ModelError::UnrecognizedFormat)
    ));
}

#[test]
fn a_legacy_activation_tag_other_than_none_or_some_is_unrecognized() {
    let mut bytes = raw_bytes(vec![raw_layer(3, 2), raw_layer(2, 3)]);
    // The file ends with the activation's Option tag, 0 for None.
    *bytes.last_mut().unwrap() = 2;

    assert!(matches!(
        read_model::<2, 2>(bytes.as_slice()),
        Err(ModelError::UnrecognizedFormat)
    ));
}

#[test]
fn serde_round_trips_through_a_plain_readable_form() {
    let mut original = NeuralNetwork::<2, 1>::new(&[]);
    original.set_layer_weights(1, &[[0.5, -0.25]]);
    original.set_layer_biases(1, &[0.1]);
    original.set_output_activation(ActivationFunction::Tanh);

    let json = serde_json::to_string(&original).unwrap();
    assert_eq!(
        json,
        r#"{"layers":[{"activation":"Tanh","weights":[[0.5,-0.25]],"biases":[0.1]}]}"#
    );

    let loaded: NeuralNetwork<2, 1> = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded, original);
}

/// Validation lives in `Deserialize` itself, so other serde entry points cannot
/// skip it.
#[test]
fn deserializing_directly_also_validates() {
    let broken_chain = r#"{"layers":[
        {"activation":"Tanh","weights":[[1,2],[3,4],[5,6]],"biases":[0,0,0]},
        {"activation":"Tanh","weights":[[1,2,3,4],[1,2,3,4]],"biases":[0,0]}
    ]}"#;
    let wrong_output = r#"{"layers":[
        {"activation":"Tanh","weights":[[1,2]],"biases":[0]}
    ]}"#;

    let error = serde_json::from_str::<NeuralNetwork<2, 2>>(broken_chain).unwrap_err();
    assert!(error.to_string().contains("layer 2"), "unexpected error: {error}");
    assert!(serde_json::from_str::<NeuralNetwork<2, 2>>(wrong_output).is_err());
}

#[test]
fn deserializing_rejects_a_layer_without_neurons() {
    let empty_first = r#"{"layers":[
        {"activation":"Tanh","weights":[],"biases":[]}
    ]}"#;
    let empty_hidden = r#"{"layers":[
        {"activation":"Tanh","weights":[[1,2],[3,4]],"biases":[0,0]},
        {"activation":"Tanh","weights":[],"biases":[]},
        {"activation":"Tanh","weights":[[1,2],[3,4]],"biases":[0,0]}
    ]}"#;

    for (json, layer) in [(empty_first, 1), (empty_hidden, 2)] {
        let error = serde_json::from_str::<NeuralNetwork<2, 2>>(json).unwrap_err();
        assert!(
            error.to_string().contains(&format!("layer {layer} of the model is inconsistent: it has no neurons")),
            "unexpected error: {error}"
        );
    }
}

#[test]
fn loading_a_missing_file_returns_an_error_instead_of_panicking() {
    let missing = std::env::temp_dir().join("only-brain-does-not-exist.bin");
    let _ = fs::remove_file(&missing);

    assert!(load_model::<2, 2>(&missing).is_err());
}

#[test]
fn loading_a_file_that_is_not_a_model_reports_an_unrecognized_format() {
    let temp = TempModelPath::new("garbage");
    fs::write(temp.path(), b"this is not a bincode-encoded model").unwrap();

    assert!(matches!(
        load_model::<2, 2>(temp.path()),
        Err(ModelError::UnrecognizedFormat)
    ));
}

#[test]
fn loading_an_empty_file_reports_an_unrecognized_format() {
    let temp = TempModelPath::new("empty");
    fs::write(temp.path(), b"").unwrap();

    assert!(matches!(
        load_model::<2, 2>(temp.path()),
        Err(ModelError::UnrecognizedFormat)
    ));
}

fn to_bytes<const IN: usize, const OUT: usize>(nn: &NeuralNetwork<IN, OUT>) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_model(nn, &mut bytes).expect("writing to memory should succeed");
    bytes
}

#[test]
fn a_saved_model_starts_with_the_magic_and_the_format_version() {
    let bytes = to_bytes(&sample_network());

    assert_eq!(&bytes[..8], b"ONLYBRN\0");
    assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), MODEL_FORMAT_VERSION);
}

#[test]
fn a_model_from_a_newer_format_version_is_reported_as_unsupported() {
    let mut bytes = to_bytes(&sample_network());
    bytes[8..10].copy_from_slice(&(MODEL_FORMAT_VERSION + 1).to_le_bytes());

    match read_model::<2, 2>(bytes.as_slice()) {
        Err(ModelError::UnsupportedVersion { found, newest }) => {
            assert_eq!(found, MODEL_FORMAT_VERSION + 1);
            assert_eq!(newest, MODEL_FORMAT_VERSION);
        }
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

#[test]
fn a_truncated_model_is_reported_as_malformed() {
    let bytes = to_bytes(&sample_network());

    for length in [9, 12, 20, bytes.len() - 1] {
        assert!(
            matches!(
                read_model::<2, 2>(&bytes[..length]),
                Err(ModelError::Malformed { .. })
            ),
            "a model cut at {length} bytes should be malformed"
        );
    }
}

#[test]
fn an_unknown_activation_code_is_reported_as_malformed() {
    let mut bytes = to_bytes(&sample_network());
    // magic (8) + version (2) + inputs (4) + layers (4) + first layer's neurons (4)
    bytes[22] = 200;

    assert!(matches!(
        read_model::<2, 2>(bytes.as_slice()),
        Err(ModelError::Malformed { .. })
    ));
}

#[test]
fn a_corrupted_neuron_count_fails_instead_of_exhausting_memory() {
    let mut bytes = to_bytes(&sample_network());
    bytes[18..22].copy_from_slice(&u32::MAX.to_le_bytes());

    assert!(read_model::<2, 2>(bytes.as_slice()).is_err());
}

#[test]
fn a_layer_with_no_neurons_is_reported_as_malformed() {
    let zero_neurons = {
        let mut bytes = to_bytes(&sample_network());
        // magic (8) + version (2) + inputs (4) + layers (4)
        bytes[18..22].copy_from_slice(&0u32.to_le_bytes());
        bytes
    };
    let zero_inputs = {
        let mut bytes = to_bytes(&sample_network());
        bytes[10..14].copy_from_slice(&0u32.to_le_bytes());
        bytes
    };

    for bytes in [zero_neurons, zero_inputs] {
        match read_model::<2, 2>(bytes.as_slice()) {
            Err(ModelError::Malformed { reason }) => assert_eq!(reason, "a layer has no neurons"),
            other => panic!("expected a malformed model, got {other:?}"),
        }
    }
}

/// A reader that fails the way a disk or a network can, rather than by ending early.
struct FailingReader;

impl std::io::Read for FailingReader {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("the device went away"))
    }
}

#[test]
fn a_read_error_is_reported_as_io_rather_than_as_a_corrupted_model() {
    let bytes = to_bytes(&sample_network());
    // The magic and version arrive, then the reader fails.
    let reader = std::io::Read::chain(&bytes[..10], FailingReader);

    match read_model::<2, 2>(reader) {
        Err(ModelError::Io(error)) => assert_eq!(error.kind(), std::io::ErrorKind::Other),
        other => panic!("expected an I/O error, got {other:?}"),
    }
}

#[test]
fn a_file_with_data_after_the_model_is_reported_as_malformed() {
    let temp = TempModelPath::new("trailing");
    let mut bytes = to_bytes(&sample_network());
    bytes.push(0);
    fs::write(temp.path(), bytes).unwrap();

    assert!(matches!(
        load_model::<2, 2>(temp.path()),
        Err(ModelError::Malformed { .. })
    ));
}

#[test]
fn several_models_can_be_read_from_one_stream() {
    let first = sample_network();
    let second = NeuralNetwork::<3, 1>::new(&[4]);
    let mut stream = to_bytes(&first);
    stream.extend(to_bytes(&second));

    let mut reader = stream.as_slice();
    let read_first: NeuralNetwork<2, 2> = read_model(&mut reader).unwrap();
    let read_second: NeuralNetwork<3, 1> = read_model(&mut reader).unwrap();

    assert_eq!(read_first, first);
    assert_eq!(read_second, second);
    assert!(reader.is_empty());
}

#[test]
fn the_saved_weights_are_the_flat_parameter_view() {
    let nn = sample_network();
    let bytes = to_bytes(&nn);

    // After the 18-byte header, each layer is 5 bytes of shape then its parameters.
    let mut floats = Vec::new();
    let mut offset = 18;
    for layer in 1..nn.num_layers() {
        offset += 5;
        let count = nn.layer_size(layer) * (nn.layer_size(layer - 1) + 1);
        let (chunks, rest) = bytes[offset..offset + 8 * count].as_chunks::<8>();
        assert!(rest.is_empty());
        floats.extend(chunks.iter().map(|chunk| f64::from_le_bytes(*chunk)));
        offset += 8 * count;
    }

    assert_eq!(offset, bytes.len());
    assert_eq!(floats, nn.parameters());
}

/// The network stored in `tests/fixtures/v1_model.bin`.
fn v1_fixture_network() -> NeuralNetwork<2, 2> {
    let mut nn = sample_network();
    nn.set_layer_activation(1, ActivationFunction::Tanh);
    nn.set_output_activation(ActivationFunction::Identity);
    nn
}

/// Saved models must keep loading, and saving must keep producing the same bytes
/// until the format version changes. Regenerate the fixture only together with a new
/// format version, by running this test with `ONLY_BRAIN_BLESS=1`.
#[test]
fn the_version_1_format_is_stable() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_model.bin");
    let expected = v1_fixture_network();

    if std::env::var_os("ONLY_BRAIN_BLESS").is_some() {
        dump_model(&expected, &path).expect("writing the fixture should succeed");
    }

    let loaded: NeuralNetwork<2, 2> = load_model(&path).expect("version 1 model should load");
    assert_eq!(loaded, expected);
    assert_eq!(to_bytes(&expected), fs::read(&path).unwrap());
}

/// Written by 0.1.x (`examples/neural_network.rs`). The format must stay loadable.
#[test]
fn a_model_written_by_an_earlier_version_still_loads() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0_1_model.bin");

    let loaded: NeuralNetwork<2, 2> = load_model(path).expect("0.1.x model should load");

    assert_eq!(loaded.layer_size(1), 3);
    assert_eq!(loaded.layer_activation(1), ActivationFunction::BinaryStep);
    assert_eq!(loaded.output_activation(), ActivationFunction::BinaryStep);
    assert_eq!(loaded.weight(1, 0, 0), 0.99);
    assert_eq!(loaded.weight(2, 1, 2), 0.4);
}

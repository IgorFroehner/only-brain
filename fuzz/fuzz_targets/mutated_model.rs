//! Saves a valid network of a shape chosen by the input, corrupts some of its bytes
//! and truncates it as the rest of the input says, then reads it back. `read_model`
//! must return an error or a network, never panic. This reaches deep into the version
//! 1 reader faster than arbitrary bytes, which rarely get past the header.

#![no_main]

use libfuzzer_sys::fuzz_target;
use only_brain::{ActivationFunction, NeuralNetwork, read_model, write_model};

fuzz_target!(|data: &[u8]| {
    let [shape, truncate_low, truncate_high, edits @ ..] = data else {
        return;
    };

    // Up to three hidden layers of 1 to 4 neurons: the depth, then two bits per layer.
    let depth = usize::from(shape & 0b11);
    let hidden: Vec<usize> = (0..depth)
        .map(|layer| usize::from((shape >> (2 + 2 * layer)) & 0b11) + 1)
        .collect();
    let parameters: Vec<f64> = (0..NeuralNetwork::<2, 2>::parameter_count_for(&hidden))
        .map(|index| index as f64 * 0.25 - 1.0)
        .collect();
    let mut network = NeuralNetwork::<2, 2>::from_parameters(&hidden, &parameters);
    let functions = ActivationFunction::ALL;
    for layer in 1..network.num_layers() {
        let activation = functions[(layer + usize::from(*shape)) % functions.len()];
        network.set_layer_activation(layer, activation);
    }

    let mut bytes = Vec::new();
    write_model(&network, &mut bytes).expect("a network saves to memory");

    // Each edit is a little-endian u16 byte offset and a value to XOR there.
    for edit in edits.chunks_exact(3) {
        let offset = usize::from(u16::from_le_bytes([edit[0], edit[1]])) % bytes.len();
        bytes[offset] ^= edit[2];
    }
    // A truncation of 0 keeps every byte; anything else cuts the data short.
    let truncate = usize::from(u16::from_le_bytes([*truncate_low, *truncate_high]));
    if truncate != 0 {
        bytes.truncate(truncate % bytes.len());
    }

    let _ = read_model::<2, 2>(bytes.as_slice());
});

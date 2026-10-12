//! Reads models written before 0.3, which have no magic or version.
//!
//! Those files are the bincode 1 encoding (fixed-width little-endian integers, `u64`
//! lengths) of the network as it was stored then:
//!
//! ```text
//! layers: u64 count, then for each layer:
//!     size:    u64
//!     weights: nalgebra DMatrix = u64 length, f64 values in column-major order,
//!              u64 rows, u64 columns
//!     bias:    nalgebra DVector = u64 length, f64 values, u64 rows
//! activation_function: u8 Option tag, then a u32 variant index if present
//! ```
//!
//! The format is frozen, so it is decoded by hand here rather than by keeping bincode 1
//! and the old nalgebra serde layout as dependencies.

use std::io::Read;

use super::Decoder;
use crate::neural_network::{LayerData, NetworkData};
use crate::{ActivationFunction, ModelError};

/// Reads the rest of a legacy model whose first eight bytes, the layer count, have
/// already been read.
pub(super) fn read<R: Read>(layer_count: u64, decoder: &mut Decoder<R>) -> Result<NetworkData, ModelError> {
    read_network(layer_count, decoder).map_err(|error| match error {
        // Without a magic to go by, data that does not parse is most likely not a model.
        ModelError::Malformed { .. } => ModelError::UnrecognizedFormat,
        error => error,
    })
}

fn read_network<R: Read>(layer_count: u64, decoder: &mut Decoder<R>) -> Result<NetworkData, ModelError> {
    let mut layers = Vec::new();
    for index in 0..layer_count {
        layers.push(read_layer(decoder, index as usize + 1)?);
    }

    let activation = match decoder.u8()? {
        0 => ActivationFunction::default(),
        // The four functions that existed before 0.3, by their variant index then.
        // This table is frozen: functions added later never appear in these files.
        1 => match decoder.u32()? {
            0 => ActivationFunction::Sigmoid,
            1 => ActivationFunction::Tanh,
            2 => ActivationFunction::ReLU,
            3 => ActivationFunction::BinaryStep,
            _ => return Err(ModelError::UnrecognizedFormat),
        },
        _ => return Err(ModelError::UnrecognizedFormat),
    };
    for layer in &mut layers {
        layer.activation = activation;
    }

    Ok(NetworkData { layers })
}

fn read_layer<R: Read>(decoder: &mut Decoder<R>, layer: usize) -> Result<LayerData, ModelError> {
    let size = decoder.u64()?;

    let columns_first = read_floats(decoder)?;
    let rows = decoder.u64()?;
    let columns = decoder.u64()?;
    if rows.checked_mul(columns) != Some(columns_first.len() as u64) || (rows > 0 && columns == 0) {
        return Err(ModelError::UnrecognizedFormat);
    }

    let biases = read_floats(decoder)?;
    if decoder.u64()? != biases.len() as u64 {
        return Err(ModelError::UnrecognizedFormat);
    }

    if size != rows {
        return Err(ModelError::InconsistentLayer {
            layer,
            reason: "its size does not match its number of weight rows",
        });
    }

    // Both counts are bounded by the values actually read, so these fit in memory.
    let (rows, columns) = (rows as usize, columns as usize);
    let weights = (0..rows)
        .map(|row| (0..columns).map(|column| columns_first[column * rows + row]).collect())
        .collect();

    Ok(LayerData {
        activation: ActivationFunction::default(),
        weights,
        biases,
    })
}

/// A `u64` length followed by that many floats.
fn read_floats<R: Read>(decoder: &mut Decoder<R>) -> Result<Vec<f64>, ModelError> {
    let length = decoder.u64()?;
    let length = usize::try_from(length).map_err(|_| ModelError::UnrecognizedFormat)?;
    decoder.f64s(length)
}

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::neural_network::NetworkData;
use crate::NeuralNetwork;

/// Something that went wrong while saving or loading a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The file could not be read or written.
    #[error("could not access the model file")]
    Io(#[from] std::io::Error),

    /// The bytes were not a model, or were written by an incompatible version.
    #[error("model file is malformed or was not produced by this library")]
    Serialization(#[from] bincode::Error),

    /// The stored network has a different shape than the type asked for.
    #[error("model has {found} {end} neurons but {expected} were expected")]
    DimensionMismatch {
        /// Which end of the network mismatched.
        end: &'static str,
        /// The width required by the requested type.
        expected: usize,
        /// The width actually found in the file.
        found: usize,
    },

    /// The file decoded to a network with no layers at all.
    #[error("model file contains a network with no layers")]
    EmptyNetwork,

    /// A stored layer does not fit with itself or with the layer before it, which means
    /// the file was corrupted or edited by hand.
    #[error("layer {layer} of the model is inconsistent: {reason}")]
    InconsistentLayer {
        /// The layer number, counting the input layer as 0.
        layer: usize,
        /// What disagreed.
        reason: &'static str,
    },
}

/// Writes a model to `path`.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file cannot be written, or
/// [`ModelError::Serialization`] if encoding fails.
pub fn dump_model<const IN: usize, const OUT: usize>(
    model: &NeuralNetwork<IN, OUT>,
    path: impl AsRef<Path>,
) -> Result<(), ModelError> {
    let encoded = bincode::serialize(model)?;

    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(&encoded)?;
    file.flush()?;

    Ok(())
}

/// Reads a model from `path`, checking that its shape matches `IN` and `OUT`.
///
/// The input and output widths are recovered from the stored weight matrices, so
/// [`dump_model`] writes no extra data and files produced by earlier versions still
/// load -- provided their dimensions match the type you ask for. Every layer is also
/// checked against its neighbours, so a corrupted file is reported here rather than
/// panicking later in [`NeuralNetwork::feed_forward`].
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file cannot be read, [`ModelError::Serialization`]
/// if it is not a valid model, [`ModelError::DimensionMismatch`] if the stored network
/// does not fit `NeuralNetwork<IN, OUT>`, and [`ModelError::EmptyNetwork`] or
/// [`ModelError::InconsistentLayer`] if its layers do not form a network at all.
///
/// # Example
///
/// ```no_run
/// # use only_brain::{load_model, NeuralNetwork};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let nn: NeuralNetwork<2, 1> = load_model("model.bin")?;
/// # Ok(())
/// # }
/// ```
pub fn load_model<const IN: usize, const OUT: usize>(
    path: impl AsRef<Path>,
) -> Result<NeuralNetwork<IN, OUT>, ModelError> {
    let file = BufReader::new(File::open(path)?);
    // Decoding the unchecked form keeps the typed error; deserializing the network
    // directly would fold it into a bincode message.
    let data: NetworkData = bincode::deserialize_from(file)?;

    NeuralNetwork::try_from(data)
}

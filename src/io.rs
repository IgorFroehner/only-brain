//! Saving and loading networks in the model file format.
//!
//! # Format
//!
//! Every file starts with an 8-byte magic, `ONLYBRN\0`, and a format version. Version
//! 1, the one written today, continues as follows. Every number is little-endian.
//!
//! | Field | Type | Meaning |
//! |---|---|---|
//! | magic | 8 bytes | `ONLYBRN\0` |
//! | version | `u16` | `1` |
//! | inputs | `u32` | number of input neurons |
//! | layers | `u32` | number of layers after the input layer |
//!
//! Then, for each layer after the input layer:
//!
//! | Field | Type | Meaning |
//! |---|---|---|
//! | neurons | `u32` | number of neurons in this layer |
//! | activation | `u8` | 0 sigmoid, 1 tanh, 2 ReLU, 3 binary step, 4 identity, 5 leaky ReLU, 6 ELU, 7 softplus, 8 SiLU |
//! | weights | `f64` × neurons × previous layer's neurons | one row per neuron |
//! | biases | `f64` × neurons | one per neuron |
//!
//! The weights and biases, read in file order, are exactly the network's [flat
//! parameter view](crate::NeuralNetwork#flat-parameter-view).
//!
//! Files written before 0.3 have no magic: they are the bincode 1 encoding of the
//! network, with one activation function for the whole network. They still load, and
//! every layer gets that activation function.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::neural_network::{LayerData, NetworkData};
use crate::{ActivationFunction, NeuralNetwork};

mod legacy;

/// The bytes every model file written since 0.3 starts with.
const MAGIC: [u8; 8] = *b"ONLYBRN\0";

/// The model format version [`dump_model`] and [`write_model`] write. [`load_model`] and
/// [`read_model`] read this version and every earlier one.
pub const MODEL_FORMAT_VERSION: u16 = 1;

/// Something that went wrong while saving or loading a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The file could not be read or written.
    #[error("could not access the model file")]
    Io(#[from] std::io::Error),

    /// The data is not a model written by this library.
    #[error("data is not a model produced by this library")]
    UnrecognizedFormat,

    /// The model was written by a newer version of this library, in a format this
    /// version does not know.
    #[error("model format version {found} is not supported, the newest known is {newest}")]
    UnsupportedVersion {
        /// The version stored in the file.
        found: u16,
        /// The newest version this library reads, [`MODEL_FORMAT_VERSION`].
        newest: u16,
    },

    /// The data starts like a model but is truncated or corrupted.
    #[error("model data is corrupted: {reason}")]
    Malformed {
        /// What was wrong.
        reason: &'static str,
    },

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

/// Writes a model to `path`, in the newest format version.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file cannot be written.
///
/// # Example
///
/// ```no_run
/// # use only_brain::{dump_model, NeuralNetwork};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let nn = NeuralNetwork::<2, 1>::new(&[2]);
/// dump_model(&nn, "model.bin")?;
/// # Ok(())
/// # }
/// ```
pub fn dump_model<const IN: usize, const OUT: usize>(
    model: &NeuralNetwork<IN, OUT>,
    path: impl AsRef<Path>,
) -> Result<(), ModelError> {
    let mut file = BufWriter::new(File::create(path)?);
    write_model(model, &mut file)?;
    file.flush()?;

    Ok(())
}

/// Reads a model from `path`, checking that its shape matches `IN` and `OUT`.
///
/// Files written by every earlier version of this library load too. Every layer is
/// checked against its neighbours, so a corrupted file is reported here rather than
/// panicking later in [`NeuralNetwork::feed_forward`], and the file must end where the
/// model does.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if the file cannot be read,
/// [`ModelError::UnrecognizedFormat`] if it is not a model,
/// [`ModelError::UnsupportedVersion`] if a newer version of this library wrote it,
/// [`ModelError::Malformed`] if it is truncated or corrupted,
/// [`ModelError::DimensionMismatch`] if the stored network does not fit
/// `NeuralNetwork<IN, OUT>`, and [`ModelError::EmptyNetwork`] or
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
    let mut file = BufReader::new(File::open(path)?);
    let model = read_model(&mut file)?;

    if file.read(&mut [0])? != 0 {
        return Err(ModelError::Malformed {
            reason: "the file continues after the model",
        });
    }

    Ok(model)
}

/// Writes a model to any writer, in the newest format version.
///
/// The model is encoded in memory and written with a single `write_all`, so the writer
/// does not need to be buffered. Use it with a `Vec<u8>` to keep a model in memory,
/// for example to store it in a database.
///
/// # Errors
///
/// Returns [`ModelError::Io`] if writing fails.
///
/// # Example
///
/// ```
/// # use only_brain::{read_model, write_model, NeuralNetwork};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let nn = NeuralNetwork::<2, 1>::new(&[2]);
///
/// let mut bytes = Vec::new();
/// write_model(&nn, &mut bytes)?;
///
/// let copy: NeuralNetwork<2, 1> = read_model(bytes.as_slice())?;
/// assert_eq!(copy, nn);
/// # Ok(())
/// # }
/// ```
pub fn write_model<const IN: usize, const OUT: usize>(
    model: &NeuralNetwork<IN, OUT>,
    mut writer: impl Write,
) -> Result<(), ModelError> {
    let layers = model.layers();
    let mut bytes = Vec::with_capacity(MAGIC.len() + 10 + 8 * model.parameter_count() + 5 * layers.len());

    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&MODEL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&format_count(IN).to_le_bytes());
    bytes.extend_from_slice(&format_count(layers.len()).to_le_bytes());

    for layer in layers {
        bytes.extend_from_slice(&format_count(layer.size()).to_le_bytes());
        bytes.push(layer.activation().code());
        // The weights row by row, then the biases: the flat parameter view, which is
        // also the order the layer stores its weights in.
        for value in layer.flat_weights() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in layer.biases() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    writer.write_all(&bytes)?;
    Ok(())
}

/// Reads one model from any reader, checking that its shape matches `IN` and `OUT`.
///
/// Reading stops where the model ends, so several models can be read one after the
/// other from the same stream. Wrap slow readers such as files in a
/// [`std::io::BufReader`].
///
/// # Errors
///
/// The same as [`load_model`], except that data after the model is left unread rather
/// than reported.
pub fn read_model<const IN: usize, const OUT: usize>(
    reader: impl Read,
) -> Result<NeuralNetwork<IN, OUT>, ModelError> {
    let mut decoder = Decoder::new(reader);

    let start: [u8; 8] = decoder.bytes().map_err(|_| ModelError::UnrecognizedFormat)?;
    let data = if start == MAGIC {
        match decoder.u16()? {
            1 => read_v1(&mut decoder)?,
            found => {
                return Err(ModelError::UnsupportedVersion {
                    found,
                    newest: MODEL_FORMAT_VERSION,
                })
            }
        }
    } else {
        legacy::read(u64::from_le_bytes(start), &mut decoder)?
    };

    NeuralNetwork::try_from(data)
}

/// Reads the rest of a version 1 model, after the magic and the version.
fn read_v1<R: Read>(decoder: &mut Decoder<R>) -> Result<NetworkData, ModelError> {
    let mut inputs = decoder.u32()? as usize;
    let layer_count = decoder.u32()?;

    let mut layers = Vec::new();
    for _ in 0..layer_count {
        let neurons = decoder.u32()? as usize;
        // Every later read then consumes data, so a corrupted count runs out of input
        // instead of looping without end.
        if neurons == 0 || inputs == 0 {
            return Err(ModelError::Malformed {
                reason: "a layer has no neurons",
            });
        }
        let activation = ActivationFunction::from_code(decoder.u8()?).ok_or(ModelError::Malformed {
            reason: "unknown activation function code",
        })?;

        let mut weights = Vec::new();
        for _ in 0..neurons {
            weights.push(decoder.f64s(inputs)?);
        }
        let biases = decoder.f64s(neurons)?;

        layers.push(LayerData {
            activation,
            weights,
            biases,
        });
        inputs = neurons;
    }

    Ok(NetworkData { layers })
}

/// A count as the format stores it.
fn format_count(count: usize) -> u32 {
    u32::try_from(count).expect("a layer cannot have more than u32::MAX neurons")
}

/// Reads exactly enough bytes to fill `bytes`, reporting a short read as a malformed
/// model.
fn fill(reader: &mut impl Read, bytes: &mut [u8]) -> Result<(), ModelError> {
    reader.read_exact(bytes).map_err(|error| match error.kind() {
        std::io::ErrorKind::UnexpectedEof => ModelError::Malformed {
            reason: "the data ends before the model does",
        },
        _ => ModelError::Io(error),
    })
}

/// How many floats [`Decoder::f64s`] reads per chunk.
const DECODE_CHUNK: usize = 512;

/// Reads little-endian values, reporting a short read as a malformed model.
struct Decoder<R> {
    reader: R,
    /// Scratch for bulk reads, so each read does not initialise its own buffer.
    buffer: Box<[u8; 8 * DECODE_CHUNK]>,
}

impl<R: Read> Decoder<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: Box::new([0; 8 * DECODE_CHUNK]),
        }
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let mut bytes = [0; N];
        fill(&mut self.reader, &mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(u8::from_le_bytes(self.bytes()?))
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// Reads `count` floats, in bulk but one bounded chunk at a time. The count comes
    /// from the data, so memory is reserved as values arrive rather than up front, and a
    /// corrupted count fails at the end of the data instead of exhausting memory.
    fn f64s(&mut self, count: usize) -> Result<Vec<f64>, ModelError> {
        let mut values = Vec::with_capacity(count.min(8 * DECODE_CHUNK));
        let mut remaining = count;
        while remaining > 0 {
            let chunk = remaining.min(DECODE_CHUNK);
            let bytes = &mut self.buffer[..8 * chunk];
            fill(&mut self.reader, bytes)?;
            // The slice holds 8 * chunk bytes, so there is no remainder.
            let (chunks, _) = bytes.as_chunks::<8>();
            values.extend(chunks.iter().map(|value| f64::from_le_bytes(*value)));
            remaining -= chunk;
        }
        Ok(values)
    }
}

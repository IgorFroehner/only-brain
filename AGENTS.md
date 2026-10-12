# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex and others) when
working with code in this repository.

## What this is

`only-brain` is a Rust library crate: a feed-forward neural network "without the
learning part". It exposes a network's weights and biases so users bring their own
training. The main consumer is neuroevolution: the flat parameter view exists so a
genetic algorithm (the sibling `evolutionary` crate, see `ideas/racelearn.md`) can
treat a network as a `Vec<f64>` genome. Keep that use case fast and the API plain.

## Commands

```sh
cargo test --all-targets                        # unit + integration tests, builds examples and benches
cargo test --doc                                # doctests, including README.md (see below)
cargo test --test model_io                      # one integration test file
cargo test --test model_io a_legacy_file_loads  # one integration test
cargo test feed_forward                         # unit tests matching a name
cargo clippy --all-targets -- -D warnings       # CI fails on any warning
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps  # CI fails on broken intra-doc links
cargo bench                                     # criterion; `cargo bench -- feed_forward` for one group
cargo run --example xor_evolution               # end-to-end check of the flat parameter view, run in CI
cargo run --example iris_backprop               # end-to-end check of backprop through the public API, run in CI
cargo publish --dry-run                         # CI packages the crate on every push
```

CI (`.github/workflows/ci.yml`) runs exactly these. Run the test, clippy and doc
commands before considering a change done.

## Things that are easy to break

- **README.md is compiled as doctests.** `src/lib.rs` includes it via
  `#[doc = include_str!("../README.md")]` under `cfg(doctest)`. Every fenced block
  without a language tag is treated as Rust and must compile. Tag shell or other
  non-Rust blocks (` ```sh `, ` ```text `). Dev-dependencies such as `serde_json`
  are available there.
- **The flat parameter order is a contract.** Layer by layer, each layer's weights
  row by row (one row per neuron, columns in previous-layer order), then biases. It
  is documented on `NeuralNetwork`, it is the genome layout for GA users, and it is
  byte-for-byte the model file layout. Never change it.
- **The model file format is frozen per version.** `src/io.rs` documents it: magic
  `ONLYBRN\0`, `u16` version, then layers. `ActivationFunction::code()` values are
  part of the format: new variants get new codes appended, existing ones never move,
  and `ActivationFunction::ALL` must list every variant. Adding a format version means
  keeping `read_v1` and every older reader. `tests/fixtures/v0_1_model.bin` and
  `v1_model.bin` guard this.
- **Pre-0.3 files have no magic.** `src/io/legacy.rs` decodes the old bincode 1 /
  nalgebra layout by hand (column-major weights, network-wide activation). The
  `bincode` dev-dependency exists only so tests can write such files.
- **Published crate contents are an allowlist.** `Cargo.toml` `include` ships only
  `src`, `Cargo.toml`, `README.md`, `CHANGELOG.md` and `LICENSE`. Examples, tests
  and benches are not published, so README links to them must be absolute GitHub
  URLs, not relative paths.

## Architecture

All modules are private and re-exported flat from `src/lib.rs`, so the public API is
`only_brain::{NeuralNetwork, Trace, ActivationFunction, softmax, argmax, Init,
Perceptron, dump_model, load_model, write_model, read_model, ModelError,
MODEL_FORMAT_VERSION}`, plus the deprecated `BVector` and `bvector!`. `src/lib.rs`
re-exports `activation_functions` by name: the plain `fn(f64) -> f64` functions there
are `pub(crate)`, and a public helper function added to that module must also be
listed in that `pub use`.

- `src/neural_network.rs`: `NeuralNetwork<IN, OUT>` is a `Vec<Layer>`. The input
  layer is **layer 0** in every public index, has no weights, and is not stored.
  Public layer `n` maps to `self.layers[n - 1]`, and layer 0 panics with
  `"Invalid layer index"`. `num_layers()` counts the input layer. Setters panic on
  shape mismatch with messages the unit tests assert on (`"Incompatible weights
  matrix size"`, `"Incompatible biases vector size"`, `"at least one neuron"`).
  `Trace`, returned by `feed_forward_trace`, lives here too and uses the same layer
  numbering: `outputs(0)` is the inputs, `pre_activations(0)` panics.
- `src/layer.rs`: `Layer` holds a nalgebra `DMatrix` (one row per neuron,
  column-major in memory) and a `DVector` of biases. nalgebra never crosses the
  public boundary: every public method takes and returns arrays, slices or `Vec`s.
  `forward_into` is the allocation-free hot path that `feed_forward` calls once per
  layer with a single scratch buffer per call (on the stack when the widest layer
  has at most 16 neurons, which a benchmark showed beats a larger stack buffer).
  `forward_batch_into` is its `gemm` counterpart behind `feed_forward_batch`, one
  column per sample. `pre_activations_into` computes `W·x + b` without the
  activation, for `feed_forward_trace`. Both it and `forward_into` go through the
  private `weighted_inputs_into`, the one copy of the single-sample `gemv` (`W·x`,
  no biases). That helper is `#[inline(always)]` because a plain `#[inline]` was not
  honoured and the extra call slowed the 4-6-2 benchmark. The biases are added
  after it: `forward_into` and `forward_batch_into` add them in the same loop as the
  activation (`add_bias_and_activate`, below), `pre_activations_into` on their own.
  Each bias is added to the same product either way, so the trace stays
  bit-identical to `feed_forward`. `extend_parameters` / `take_parameters` define
  the flat order.
- **`NetworkData` / `LayerData` are the single validation funnel.** They are the
  serde form of a network (`{"layers": [{"activation", "weights", "biases"}]}`).
  `NeuralNetwork` is `#[serde(into = "NetworkData", try_from = "NetworkData")]`, and
  both file readers (`read_v1`, `legacy::read`) produce a `NetworkData` and go through
  the same `TryFrom`. That is where layers are checked against each other and against
  `IN`/`OUT`. Any new way of building a network from untrusted data should go
  through it rather than re-implementing the checks.
- `src/io.rs`: file format, `ModelError` (`thiserror`, `#[non_exhaustive]`), the
  `Decoder` that reads little-endian values in bounded chunks so a corrupted length
  cannot exhaust memory. `load_model` additionally requires the file to end where the
  model ends; `read_model` leaves trailing data unread so several models can share a
  stream.
- `src/activation_functions.rs`: `ActivationFunction` enum (`#[non_exhaustive]`,
  default `Sigmoid`), the crate-private `fn(f64) -> f64` behind each (the public way
  is `ActivationFunction::apply`), the frozen file codes, and the public `softmax`
  and `argmax` helpers. Softmax is deliberately not a variant: it needs the whole
  layer, and the file format stores one per-neuron function code per layer.
  `add_bias_and_activate` is the forward passes' hot loop: it matches the variant
  once per layer, with no wildcard arm so a new variant does not compile until it is
  handled, and runs a loop monomorphized for that function. Adding the bias in the
  same loop is deliberate: a separate bias pass followed by the (two-lane vectorized)
  sigmoid loop made the 4-6-2 benchmark 31% slower, most likely through failed
  store-to-load forwarding on layers narrower than 8 neurons.
- `src/init.rs`: `Init` (`#[non_exhaustive]`), the weight ranges behind
  `NeuralNetwork::new_with_init`. `Layer::random` draws from them; `Init::default()`
  is the `[-1, 1)` that `new` and `new_with_rng` have always used, and their output
  for a given seed must never change.
- `src/perceptron.rs`: a separate, compile-time-sized path, `Perceptron<N>` over
  `[f64; N]`, with its own serde plain form (`PerceptronData`). It converts into a
  `NeuralNetwork<N, 1>` with no hidden layers, which is how it is saved.
- `src/bvector.rs`: `BVector`, deprecated since 0.5.0 and kept only so old code
  compiles. Nothing else in the crate uses it.

Errors: IO and deserialization return `ModelError`; in-memory misuse (bad layer
index, wrong shape) panics, matching the "direct access" design.

## Conventions

- `CHANGELOG.md` is updated with every user-visible change, newest version first,
  with `Added` / `Changed` / `Fixed` / `Removed` sections and a **Breaking:** prefix
  on breaking entries. Releases are commits titled `release: X.Y.Z`.
- Public items carry rustdoc with `# Panics`, `# Errors` and `# Example` sections
  where applicable. Doc examples are tests.
- Tests are named as sentences describing the behaviour
  (`a_truncated_model_is_reported_as_malformed`). Unit tests live in `#[cfg(test)]`
  modules; public-API and format-compatibility tests live in `tests/model_io.rs`.
- Examples write `model.bin` to the working directory; it is gitignored.

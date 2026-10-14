# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex and others) when
working with code in this repository. Read `README.md` too: it is the user-facing
statement of the design, and it is compiled as tests.

## What this is

`only-brain` is a Rust library crate: a small feed-forward neural network **without
the learning part**. It provides fully connected layers, one activation function per
layer, a forward pass, and direct access to every weight and bias. How the weights are
found (backpropagation, a genetic algorithm, random search) is the user's business, so
the library never hides the numbers.

The main consumer is neuroevolution, where a genetic algorithm treats a whole network
as a `Vec<f64>` genome. Keep that use case fast and the API plain.

## Design principles

Every change should fit these. They come from the README and are what the API
promises.

1. **No training in the crate.** Do not add optimizers, loss functions or training
   loops. Add the *access* a training method needs instead: `feed_forward_trace`
   (per-layer pre-activations and outputs) and `ActivationFunction::derivative` exist
   so users can write backprop themselves, as `examples/iris_backprop.rs` does.
2. **Shape in the type.** `NeuralNetwork<IN, OUT>` makes a wrongly sized input or
   output a compile error. Hidden layer sizes are chosen at runtime.
3. **Plain data at the boundary.** Every public method takes and returns arrays,
   slices and `Vec`s. nalgebra is an internal detail and must never appear in the
   public API.
4. **Two views of the same network.** Layer by layer (`layer_weights`,
   `set_layer_weights`, `weight`, `set_bias`, ...) for methods that reason about
   layers, and one flat parameter list (`parameters`, `from_parameters`,
   `set_parameters`, `for_each_parameter_mut`, `map_parameters`) for black-box search.
   Both must stay consistent and cheap.
5. **Stable on disk.** A small, versioned binary format with no serialization
   dependency, plus serde for everything else. Files written by any earlier version
   keep loading.
6. **Errors vs. panics.** IO and deserialization return `ModelError`. In-memory
   misuse (bad layer index, wrong shape passed to a setter) panics, matching the
   direct-access design. Model readers must never panic on any input.

## Contracts that must not change

- **Layer numbering.** Layer 0 is the input layer in every public index. It has no
  weights and is not stored, so the first layer with weights is layer 1, and
  `num_layers()` counts the input layer. `Trace` uses the same numbering
  (`outputs(0)` is the inputs).
- **The flat parameter order.** Layer by layer, each layer's weights row by row (one
  row per neuron, one column per neuron of the previous layer), then that layer's
  biases. It is documented on `NeuralNetwork`, it is the genome layout GA users rely
  on, and it is exactly the model file layout. Never change it.
- **The model file format** is documented in `src/io.rs`: magic `ONLYBRN\0`, a `u16`
  version, then the layers. `ActivationFunction::code()` values are part of it: new
  variants get new codes appended, existing codes never move, and
  `ActivationFunction::ALL` lists every variant. A new format version keeps every
  older reader. Pre-0.3 files have no magic and are decoded by hand in
  `src/io/legacy.rs` (the `bincode` dev-dependency exists only to write such files in
  tests). `tests/fixtures/*.bin` guard all of this.
- **Seeded initialization is reproducible.** `new` and `new_with_rng` draw from
  `Init::default()` (`[-1, 1)`), and their output for a given seed must never change.

## Architecture

All modules are private and re-exported flat from `src/lib.rs`, so the public API is
`only_brain::{NeuralNetwork, Trace, ActivationFunction, softmax, argmax, Init,
Perceptron, dump_model, load_model, write_model, read_model, ModelError,
MODEL_FORMAT_VERSION}`, plus the deprecated `BVector` and `bvector!`. A new public
helper in `activation_functions.rs` must be added to its `pub use` in `src/lib.rs`.

- `src/neural_network.rs`: `NeuralNetwork<IN, OUT>`, a `Vec<Layer>`, and `Trace`.
  Public layer `n` is `self.layers[n - 1]`. Setter panic messages are asserted on by
  unit tests.
- `src/layer.rs`: `Layer`, a nalgebra weight matrix plus a bias vector, and the
  forward-pass hot paths. **Weights are stored transposed** (`inputs × neurons`, one
  column per neuron), so nalgebra's column-major memory *is* the flat row-per-neuron
  order and the flat view, in-place mutation and the file writer all work on one
  contiguous slice. Index it as `weights[(input, neuron)]`; `Layer`'s own methods
  still speak in rows per neuron.
- `src/activation_functions.rs`: `ActivationFunction` (`#[non_exhaustive]`, default
  `Sigmoid`), its frozen file codes, `derivative`, and the `softmax` / `argmax`
  helpers. Softmax is deliberately not a variant, since it needs the whole layer.
  `add_bias_and_activate` matches the variant once per layer with no wildcard arm, so
  a new variant does not compile until it is handled there.
- `NetworkData` / `LayerData` are **the single validation funnel**: the serde plain
  form (`{"layers": [{"activation", "weights", "biases"}]}`) that `NeuralNetwork`
  serializes through, and what both file readers produce before the same `TryFrom`
  checks layers against each other and against `IN` / `OUT`. Any new way of building
  a network from untrusted data goes through it.
- `src/io.rs`: the file format, `ModelError` (`thiserror`, `#[non_exhaustive]`), and
  a bounded `Decoder` so a corrupted length cannot exhaust memory. `load_model`
  requires the file to end where the model ends; `read_model` leaves trailing data so
  several models can share a stream.
- `src/init.rs`: `Init` (`Xavier`, `He`, `Uniform`), the per-layer weight ranges
  behind `new_with_init`.
- `src/perceptron.rs`: `Perceptron<N>`, a separate compile-time-sized single neuron
  with its own serde form. It converts into a `NeuralNetwork<N, 1>` with no hidden
  layers, which is how it is saved.
- `src/bvector.rs`: deprecated since 0.5.0, kept only so old code compiles.

The hot paths in `src/layer.rs` and `add_bias_and_activate` were tuned against the
benchmarks in `benches/`, and the comments there record why (stack scratch
buffers, which nalgebra product to use, adding the bias inside the activation loop).
Read them before restructuring, and benchmark any change to them.

## Commands

```sh
cargo test --all-targets                        # unit + integration tests, builds examples and benches
cargo test --doc                                # doctests, including README.md
cargo test --test model_io                      # one integration test file
cargo test --test model_io a_legacy_file_loads  # one integration test
cargo test feed_forward                         # unit tests matching a name
cargo clippy --all-targets -- -D warnings
cargo fmt --check                               # also: cargo fmt --check --manifest-path fuzz/Cargo.toml
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo bench                                     # criterion, every target; compares with the previous run
cargo bench --bench inference -- '^feed_forward' # one target, ids matching a regex
scripts/bench-compare.sh                        # main, then the working tree, and what changed
cargo run --example xor_evolution               # end-to-end check of the flat view, run in CI
cargo run --example iris_backprop               # end-to-end check of the layer view and trace, run in CI
cargo +1.88 test --all-targets                  # the MSRV
cargo +nightly fuzz run read_model              # needs cargo-fuzz; also `mutated_model`
```

CI (`.github/workflows/ci.yml`) runs all of these except the benchmarks, plus
`cargo publish --dry-run` and `cargo semver-checks` against the latest crates.io
release. Run the test, clippy, fmt and doc commands before considering a change done.

## Benchmarks

Four Criterion targets in `benches/`, sharing `benches/common/mod.rs`:

- `inference`: `feed_forward` across shapes, per activation, batched and traced;
  `apply` and `derivative`; `softmax`, `argmax`; `Perceptron`.
- `parameters`: the flat view, construction and cloning, and the layer view.
- `model_io`: the binary format and serde.
- `workloads`: a genetic algorithm generation and a backpropagation epoch, end to end.

Benchmark ids (`group/function/shape`) are what baselines are matched on, so do not
rename existing ones; add new ones instead. To measure a change, either save the
unchanged code with `cargo bench -- --save-baseline before` and compare with
`cargo bench -- --baseline-lenient before` (as often as needed; it does not
overwrite `before`), or run
`scripts/bench-compare.sh [ref] [cargo options] [-- criterion options]`, which
benchmarks `ref` (default `main`) in a temporary worktree, then the working tree,
and lists what regressed, improved, and could not be compared. Changes of a few
percent on one run are often noise on a laptop; re-run against the same baseline
before concluding. `.claude/skills/benchmarks/SKILL.md` has
the details: what each benchmark measures, reading the output, adding one.

## Things that are easy to break

- **README.md is compiled as doctests** (`src/lib.rs` includes it under
  `cfg(doctest)`). A fenced block without a language tag is Rust and must compile, so
  tag non-Rust blocks (` ```sh `, ` ```text `). Dev-dependencies like `serde_json`
  are available there.
- **Published files are an allowlist.** `Cargo.toml` `include` ships only `src`,
  `Cargo.toml`, `README.md`, `CHANGELOG.md` and `LICENSE`, so README links to
  examples, tests or `fuzz/` must be absolute GitHub URLs.
- **`fuzz/` is a separate cargo-fuzz package**, not a workspace member, with its own
  lockfile. A crash it finds is a reader bug: fix it and add the input as a
  regression test in `tests/model_io.rs`.
- **`#![warn(missing_docs)]`** plus clippy's `-D warnings` means every public item
  needs rustdoc.
- **MSRV is 1.88, edition 2024.** Raising it is a changelog entry and must also
  change the MSRV job in CI.
- **Semver is enforced.** A breaking public API change needs a minor (0.x) version
  bump, or the semver job fails.
- **Hand-aligned weight matrices** (one row per line) in code are marked
  `#[rustfmt::skip]`.

## Conventions

- `CHANGELOG.md` is updated with every user-visible change, newest version first,
  with `Added` / `Changed` / `Fixed` / `Removed` sections and a **Breaking:** prefix
  on breaking entries. Releases are commits titled `release: X.Y.Z`.
- Public items carry rustdoc with `# Panics`, `# Errors` and `# Example` sections
  where they apply. Doc examples are tests.
- Tests are named as sentences describing the behaviour
  (`a_truncated_model_is_reported_as_malformed`). Unit tests live in `#[cfg(test)]`
  modules; public-API and format-compatibility tests live in `tests/model_io.rs`.
- Examples write `model.bin` to the working directory; it is gitignored.

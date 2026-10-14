---
name: benchmarks
description: >
  How to run, compare and extend only-brain's Criterion benchmarks. Use when a change
  touches a hot path (src/layer.rs, the forward passes, the flat parameter view, the
  model reader/writer, activation functions), when asked to measure, profile or
  compare performance, check for regressions, record numbers for a CHANGELOG entry,
  or add a benchmark.
---

# Benchmarks

Criterion benchmarks in `benches/`, run locally (CI does not run them). Every number
is wall-clock on the machine that ran it, so only compare runs from the same machine.

## What exists

Four targets, each a `[[bench]]` in `Cargo.toml` with `harness = false`, sharing
`benches/common/mod.rs` (`network::<IN, OUT>(hidden)`, a seed-42 network, and
`config()`, Criterion with 1 s warm-up and 3 s measurement).

| Target | Groups | What it answers |
|---|---|---|
| `inference` (48 benches) | `feed_forward`, `feed_forward_by_activation`, `feed_forward_batch`, `feed_forward_batch_size`, `feed_forward_trace`, `activation`, `output_helpers`, `perceptron` | Forward-pass cost per shape; the stack-scratch path (widest layer, inputs included, <= 16) against the heap path; per-layer overhead (`8-8-8-8-8-8-2`); batch against a loop of single passes on three shapes, and per-input cost by batch size; the backprop forward pass; `apply`/`derivative` per variant; `softmax`, `argmax`. |
| `parameters` (26) | `parameters`, `construction`, `layer_view` | Flat view (`parameters`, `set_parameters`, `from_parameters`, `map_parameters`, `for_each_parameter_mut`, round trip); `new_with_rng`, `clone`; layer view (`layer_weights`, `set_layer_weights`, `weight`/`set_weight` one at a time, `set_layer_biases`). Each on 16-16-16-4 and 64-128-64-10. |
| `model_io` (8) | `model_io`, `serde_json` | Binary `write_model`/`read_model` on three shapes (throughput in bytes); serde through `NetworkData`. |
| `workloads` (8) | `ga_generation`, `backprop_epoch` | End to end. One GA generation over 100 individuals: copy each parent into a child, Gaussian mutation, score on a dataset. `genome` (`Vec<f64>` + `from_parameters`), `in_place` (`clone_from` + `for_each_parameter_mut`), `in_place_batch` (same, scored with `feed_forward_batch`). One SGD epoch with the `examples/iris_backprop.rs` step, on standardized Iris (4-8-3) and on synthetic 64-128-64-10 data. |

Shapes are named `IN-hidden...-OUT`. Benchmark ids are `group/function/shape`
(sometimes with a `/size`). `cargo bench -- --list` prints them all.

## Running

The full suite takes about 7 minutes; run only what the change touches.

```sh
cargo bench                                          # everything
cargo bench --bench inference                        # one target (~4 min)
cargo bench --bench inference -- '^feed_forward/'    # ids matching a regex
cargo bench --bench workloads -- ga_generation       # one group
```

### Before and after a change

Save the unchanged code under a name, then compare against that name as often as
needed:

```sh
cargo bench --bench inference -- --save-baseline before '^feed_forward'   # unchanged code
# ... make the change ...
cargo bench --bench inference -- --baseline-lenient before '^feed_forward'
```

`--baseline-lenient` never overwrites `before`, so re-running to confirm a result
compares against the same reference each time. Avoid relying on plain `cargo bench`
twice: it compares with the previous run and then overwrites it (its baseline is
`base`), so a confirming third run compares the new code with itself.

### Against a git ref

To compare against a ref without touching the working tree:

```sh
scripts/bench-compare.sh                                   # main vs the working tree
scripts/bench-compare.sh v0.5.0 -- '^feed_forward/'        # Criterion options after --
scripts/bench-compare.sh main --bench workloads            # cargo options after the ref
scripts/bench-compare.sh HEAD                              # the code against itself: noise
```

It checks the ref out into a temporary git worktree, saves its results as a baseline
named after the ref (characters other than letters, digits, `.`, `_` and `-` become
`_`, so `origin/main` is `origin_main`; `base` and `new` are refused), benchmarks the
working tree with `--baseline-lenient <name>`, and ends with three lists: regressed,
improved, and not compared (no baseline: new ids, or ids the ref does not have).
Afterwards, confirm with `cargo bench … -- --baseline-lenient <name> <filter>`; the
ref does not need to run again.

`--bench <target>` names the ref does not have are dropped for the ref's run. If
none are left, the ref runs all its targets, so pair `--bench` with a Criterion
filter when comparing against a ref from before the split (`main` until this setup
lands, `v0.5.0`). The summary parses Criterion's default output: do not pass
`--quiet` or `--format terse`.

Long runs take minutes: run them in the background and wait for them to finish,
rather than polling.

## Reading the results

```
feed_forward/16-16-16-4 time:   [163.81 ns 166.77 ns 170.22 ns]
                        change: [-8.1% -7.2% -6.3%] (p = 0.00 < 0.05)
                        Performance has improved.
```

- The middle figure is the estimate; the outer two are the 95% confidence interval.
- Criterion reports regressed or improved only when p < 0.05 and the whole
  confidence interval of the change lies beyond ±1% (the noise threshold). A +1.5%
  estimate with an interval of [+0.5%, +2.5%] is "within noise threshold".
- On a laptop, 2-5% swings between identical runs are common, especially on
  sub-100 ns benches and when other programs are busy (`scripts/bench-compare.sh
  HEAD` shows how big they are right now). Before concluding anything from a small
  change, re-run the affected filter against the same named baseline (with
  `--measurement-time 10` if it stays borderline) and check that the direction
  holds.
- `--quick` is a smoke test only: it is too noisy to decide on (one quick run showed
  +27% with p = 0.08 on unchanged code).
- `Found N outliers` is normal.
- HTML reports, with plots and the comparison against the baseline, are in
  `target/criterion/report/index.html`.

Reference points (Apple M5, 0.5.0): `feed_forward/4-6-2` ~35 ns,
`feed_forward/16-16-16-4` ~165 ns, `feed_forward/64-128-64-10` ~2.1 us,
`feed_forward/784-128-10` ~10 us, `backprop_epoch/iris/4-8-3/150` ~77 us,
`backprop_epoch/synthetic/64-128-64-10/128` ~3 ms. Numbers far from these on the
same machine mean the machine is busy, or something regressed.

## Rules for changes

- **Never rename or remove an existing id.** Baselines are matched by id, and an
  id that changes loses its history. Add new ids instead. Changing the inputs or
  the setup of an existing bench also changes what its history means, so avoid that
  too.
- **Benchmark every change to a hot path** (`src/layer.rs`, `feed_forward*`, the
  flat view, `src/io.rs`), across all shapes in `feed_forward` and
  `feed_forward_batch`, plus `workloads`. Keep a change only if it is not slower
  anywhere that matters, and say so if it trades one shape against another.
- **Record numbers** in the CHANGELOG entry, before and after, as the 0.3.1, 0.4.0
  and 0.5.0 entries do.

## Adding a benchmark

- Put it in the target whose doc comment it fits, and extend that comment.
- Build networks with `common::network` (or a fixed seed) so every run measures the
  same weights.
- Wrap inputs in `std::hint::black_box`, and return the result from the closure (or
  pass it to `black_box`), so nothing gets optimized away.
- For shapes that need different const generics, write a generic helper taking
  `&mut BenchmarkGroup<WallTime>` and a shape label, as `parameters.rs` does.
- Benches that mutate their state must start every iteration from the same place:
  overwrite the state from a fixed source inside the iteration (as `ga_generation`
  does with `clone_from`), or build it in the setup closure of `iter_batched` (as
  `backprop_epoch` does) when building it is not part of what is measured.
- Set `Throughput` when per-element or per-byte cost reads better than total time.
- With the 3 s measurement time, Criterion's default `SamplingMode::Auto` switches
  to flat sampling on its own above roughly 1.2 ms per iteration. Between about
  0.6 and 1.2 ms it stays linear and warns that it cannot collect 100 samples in
  3 s. Set `group.sampling_mode(SamplingMode::Flat)` for that one
  `bench_function` only, so the rest of the group keeps its sampling, as
  `loop_and_batch` in `inference.rs` does.
- Run `cargo clippy --all-targets -- -D warnings`: it compiles the benches.

## Gotchas

- `[lib] bench = false` in `Cargo.toml` is deliberate. Without it, `cargo bench`
  also runs the libtest harness, which rejects `--save-baseline` and `--baseline`.
  Refs older than this setup still have that problem, which is why
  `bench-compare.sh` passes `--bench '*'` unless the cargo options select targets.
- Criterion's filter is an unanchored regex: `feed_forward/` also matches
  `perceptron/feed_forward/16`. Anchor it with `^`, or use `--exact` for one id.
- `--noise-threshold 0` panics; use a small positive value.
- Up to 0.5.0 the only target was `benches/network.rs`; its ids live on unchanged
  in `inference`, `parameters` and `model_io`, so baselines saved from it still
  compare. Those runs used Criterion's default 3 s warm-up and 5 s measurement, and
  `feed_forward_batch/loop/64-128-64-10/256` was sampled linearly; both change the
  noise, not the per-iteration time being compared.

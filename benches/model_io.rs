//! Benchmarks for saving and loading networks: the binary model format, and serde
//! through `serde_json` for comparison. Throughput is in bytes of the binary file.
//!
//! Run with `cargo bench --bench model_io`.

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use only_brain::{NeuralNetwork, read_model, write_model};

mod common;
use common::network;

fn bench_model_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_io");
    model_io::<16, 4>(&mut group, &[16, 16], "16-16-16-4");
    model_io::<64, 10>(&mut group, &[128, 64], "64-128-64-10");
    model_io::<784, 10>(&mut group, &[128], "784-128-10");
    group.finish();
}

fn model_io<const IN: usize, const OUT: usize>(
    group: &mut BenchmarkGroup<WallTime>,
    hidden: &[usize],
    shape: &str,
) {
    let nn = network::<IN, OUT>(hidden);
    let mut bytes = Vec::new();
    write_model(&nn, &mut bytes).expect("writing to a Vec cannot fail");
    group.throughput(Throughput::Bytes(bytes.len() as u64));

    group.bench_function(format!("write/{shape}"), |b| {
        let mut buffer = Vec::with_capacity(bytes.len());
        b.iter(|| {
            buffer.clear();
            write_model(black_box(&nn), &mut buffer)
        })
    });

    group.bench_function(format!("read/{shape}"), |b| {
        b.iter(|| read_model::<IN, OUT>(black_box(bytes.as_slice())))
    });
}

/// The serde form goes through `NetworkData`, which copies every layer into nested
/// `Vec`s. One shape is enough to see what that costs next to the binary format.
fn bench_serde(c: &mut Criterion) {
    let mut group = c.benchmark_group("serde_json");

    let nn = network::<64, 10>(&[128, 64]);
    let json = serde_json::to_vec(&nn).expect("a network always serializes");

    group.bench_function("serialize/64-128-64-10", |b| {
        let mut buffer = Vec::with_capacity(json.len());
        b.iter(|| {
            buffer.clear();
            serde_json::to_writer(&mut buffer, black_box(&nn))
        })
    });

    group.bench_function("deserialize/64-128-64-10", |b| {
        b.iter(|| serde_json::from_slice::<NeuralNetwork<64, 10>>(black_box(&json)))
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = common::config();
    targets = bench_model_io, bench_serde
);
criterion_main!(benches);

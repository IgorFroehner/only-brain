//! Feeds arbitrary bytes to `read_model`, which must return an error rather than
//! panic, hang or exhaust memory, whatever the data. Whatever it does load must
//! round-trip: save, load back, and save to the same bytes again.
//!
//! Run it with `cargo +nightly fuzz run read_model`. To start from the format fixtures,
//! real files of both formats, as CI does:
//!
//! ```sh
//! mkdir -p fuzz/corpus/read_model
//! cargo +nightly fuzz run read_model fuzz/corpus/read_model tests/fixtures
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use only_brain::{NeuralNetwork, read_model, write_model};

fuzz_target!(|data: &[u8]| {
    let Ok(network) = read_model::<2, 2>(data) else {
        return;
    };

    // Whatever loads must save, and load back to a network that saves the same bytes.
    let mut saved = Vec::new();
    write_model(&network, &mut saved).expect("a loaded network saves");
    let reloaded: NeuralNetwork<2, 2> =
        read_model(saved.as_slice()).expect("a saved network loads");
    let mut saved_again = Vec::new();
    write_model(&reloaded, &mut saved_again).expect("a reloaded network saves");
    assert_eq!(saved, saved_again);
});

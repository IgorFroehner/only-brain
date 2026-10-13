use only_brain::{NeuralNetwork, dump_model};

fn main() {
    let mut nn = NeuralNetwork::<2, 2>::new(&[3]);

    nn.set_activation_function(only_brain::ActivationFunction::BinaryStep);

    #[rustfmt::skip]
    let first_layer_weights = [[0.1, 0.2],
                               [0.3, 0.4],
                               [0.5, 0.6]];

    nn.set_layer_weights(1, &first_layer_weights);
    nn.set_layer_biases(1, &[0.1, 0.2, 0.3]);

    #[rustfmt::skip]
    let second_layer_weights = [[0.9, 0.8, 0.7],
                                [0.6, 0.5, 0.4]];

    nn.set_layer_weights(2, &second_layer_weights);
    nn.set_layer_biases(2, &[0.1, 0.2]);

    nn.set_weight(1, 0, 0, 0.99);

    let output = nn.feed_forward(&[0.5, 0.2]);

    println!("{}", nn);
    println!("Output: {:?}", output);

    dump_model(&nn, "model.bin").expect("Failed to dump model");
}

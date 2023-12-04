use nalgebra::{dmatrix, dvector};
use only_brain::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(&vec![2, 3, 2]);

    let first_layer_weights = dmatrix![0.1, 0.2;
                                                    0.3, 0.4;
                                                    0.5, 0.6];
    nn.set_layer_weights(1, first_layer_weights);
    nn.set_layer_biases(1, dvector![0.1, 0.2, 0.3]);

    let second_layer_weights = dmatrix![0.9, 0.8, 0.7;
                                                     0.6, 0.5, 0.4];
    nn.set_layer_weights(2, second_layer_weights);
    nn.set_layer_biases(2, dvector![0.1, 0.2]);

    nn.set_weight(1, 0, 0, 0.99);

    let input = vec![0.5, 0.2];
    let output = nn.feed_forward(&input);

    nn.print();

    println!("{:?}", output);
    println!("{:?}", nn.num_layers());
    let n_layers = nn.num_layers();

    for i in 0..n_layers {
        println!("{:?}", nn.layer_size(i));
    }

    println!("{:?}", nn.get_weight(1, 0, 1));
}

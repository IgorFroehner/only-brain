use rand::{Rng, thread_rng};
use nalgebra::{DMatrix, DVector};

pub struct Layer {
    pub weights: DMatrix<f64>,
    pub bias: DVector<f64>,
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl Layer {
    pub fn forward(&self, inputs: &DVector<f64>) -> DVector<f64> {
        let outputs = &self.weights * inputs + &self.bias;
        outputs.map(|x| NeuralNetwork::activation_function(x))
    }
}

impl NeuralNetwork {
    pub fn new(layers: &Vec<usize>) -> Self {
        let mut rng = thread_rng();

        let layers = layers.iter().zip(layers.iter().skip(1)).map(|(a, b)| {
            Layer {
                weights: DMatrix::from_fn(*b, *a, |_, _| rng.gen_range(-1.0..1.0)),
                bias: DVector::from_element(*b, 0.0),
            }
        }).collect::<Vec<Layer>>();

        Self { layers }
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = DVector::from(Vec::clone(inputs));

        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }

        outputs.data.into()
    }

    pub fn activation_function(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn print(&self) {
        for layer in &self.layers {
            println!("{} {}", layer.weights, layer.bias);
        }
    }
}

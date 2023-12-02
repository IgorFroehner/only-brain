use nalgebra::{DMatrix, DVector};
use rand::distributions::Uniform;
use rand::Rng;

pub struct Layer {
    size: usize,
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

impl Layer {
    pub fn from_size<T: Rng>(neurons: usize, inputs: usize, rng: &mut T) -> Self {
        let uniform = Uniform::new(-1.0, 1.0);

        Self {
            size: neurons,
            weights: DMatrix::from_fn(neurons, inputs, |_, _| rng.sample(uniform)),
            bias: DVector::from_element(neurons, 0.0),
        }
    }

    pub fn forward(&self, inputs: &DVector<f64>, activation_func: fn(f64) -> f64) -> DVector<f64> {
        let outputs = &self.weights * inputs + &self.bias;
        outputs.map(|x| activation_func(x))
    }

    pub fn set_weight(&mut self, neuron: usize, input: usize, weight: f64) {
        self.weights[(neuron, input)] = weight;
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn set_weights(&mut self, weights: DMatrix<f64>) {
        if weights.ncols() != self.weights.ncols() || weights.nrows() != self.weights.nrows() {
            panic!("Incompatible weights matrix size");
        }
        self.weights = weights;
    }

    pub fn set_biases(&mut self, biases: DVector<f64>) {
        if biases.nrows() != self.bias.nrows() {
            panic!("Incompatible biases vector size");
        }
        self.bias = biases;
    }

    pub fn biases(&self) -> &DVector<f64> {
        &self.bias
    }

    pub fn weights(&self) -> &DMatrix<f64> {
        &self.weights
    }
}

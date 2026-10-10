use std::fmt;
use nalgebra::{DMatrix, DVector};
use rand::distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    size: usize,
    weights: DMatrix<f64>,
    bias: DVector<f64>,
}

impl Layer {
    pub fn from_size<T: Rng>(neurons: usize, inputs: usize, rng: &mut T) -> Self {
        // rand 0.9: Uniform::new(low, high) returns Result; unwrap to get distribution
        let uniform = Uniform::new(-1.0f64, 1.0f64).unwrap();

        Self {
            size: neurons,
            weights: DMatrix::from_fn(neurons, inputs, |_, _| rng.sample(uniform)),
            bias: DVector::from_element(neurons, 0.0),
        }
    }

    pub fn forward(&self, inputs: &DVector<f64>, activation_func: fn(f64) -> f64) -> DVector<f64> {
        let outputs = &self.weights * inputs + &self.bias;
        outputs.map(activation_func)
    }

    pub fn set_weight(&mut self, neuron: usize, input: usize, weight: f64) {
        self.weights[(neuron, input)] = weight;
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Replaces every weight, given one row per neuron with one entry per input.
    pub fn set_weights<R: AsRef<[f64]>>(&mut self, rows: &[R]) {
        let (neurons, inputs) = self.weights.shape();
        if rows.len() != neurons || rows.iter().any(|row| row.as_ref().len() != inputs) {
            panic!("Incompatible weights matrix size: expected {neurons} rows of {inputs} weights");
        }
        self.weights = DMatrix::from_fn(neurons, inputs, |neuron, input| rows[neuron].as_ref()[input]);
    }

    pub fn set_biases(&mut self, biases: &[f64]) {
        if biases.len() != self.bias.nrows() {
            panic!("Incompatible biases vector size: expected {} biases", self.bias.nrows());
        }
        self.bias.copy_from_slice(biases);
    }

    pub fn set_bias(&mut self, neuron: usize, bias: f64) {
        self.bias[neuron] = bias;
    }

    /// The weights as one row per neuron, each holding one weight per input.
    pub fn weight_rows(&self) -> Vec<Vec<f64>> {
        self.weights.row_iter().map(|row| row.iter().copied().collect()).collect()
    }

    /// Describes the first way this layer disagrees with itself or with the number of
    /// inputs it should receive, or returns `None` if it is consistent.
    ///
    /// Only a deserialized layer can fail this; the constructor and setters keep it true.
    pub fn inconsistency(&self, inputs: usize) -> Option<&'static str> {
        if self.weights.nrows() == 0 {
            Some("it has no neurons")
        } else if self.size != self.weights.nrows() {
            Some("its size does not match its number of weight rows")
        } else if self.bias.nrows() != self.weights.nrows() {
            Some("its number of biases does not match its number of neurons")
        } else if self.weights.ncols() != inputs {
            Some("its weight columns do not match the previous layer's size")
        } else {
            None
        }
    }

    pub fn biases(&self) -> &DVector<f64> {
        &self.bias
    }

    pub fn weights(&self) -> &DMatrix<f64> {
        &self.weights
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer Size: {}", self.size)?;
        writeln!(f, "Weights:")?;
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                write!(f, "{:0.2} ", self.weights[(i, j)])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Biases:")?;
        for bias in self.bias.iter() {
            writeln!(f, "{:0.2}", bias)?;
        }
        Ok(())
    }
}

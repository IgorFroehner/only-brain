use std::fmt;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut};
use rand::distr::Uniform;
use rand::Rng;

use crate::activation_functions::{add_bias_and_activate, ActivationFunction};

/// One layer of neurons: a weight matrix with one row per neuron and one column per
/// input, a bias per neuron, and the activation applied to every neuron.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    weights: DMatrix<f64>,
    biases: DVector<f64>,
    activation: ActivationFunction,
}

impl Layer {
    /// A layer with weights drawn uniformly from the range `init` gives for its shape,
    /// and zero biases.
    pub fn random<T: Rng>(neurons: usize, inputs: usize, activation: ActivationFunction, init: crate::Init, rng: &mut T) -> Self {
        let (low, high) = init.bounds(neurons, inputs);
        let uniform = Uniform::new(low, high).expect("the initialisation range should be finite and non-empty");

        Self {
            weights: DMatrix::from_fn(neurons, inputs, |_, _| rng.sample(uniform)),
            biases: DVector::zeros(neurons),
            activation,
        }
    }

    /// A layer whose weights and biases are all zero.
    pub fn zeros(neurons: usize, inputs: usize, activation: ActivationFunction) -> Self {
        Self {
            weights: DMatrix::zeros(neurons, inputs),
            biases: DVector::zeros(neurons),
            activation,
        }
    }

    /// Builds a layer from one weight row per neuron, or describes why the parts do
    /// not form a layer that takes `inputs` inputs.
    pub fn from_parts(
        rows: &[Vec<f64>],
        biases: &[f64],
        activation: ActivationFunction,
        inputs: usize,
    ) -> Result<Self, &'static str> {
        if rows.is_empty() {
            return Err("it has no neurons");
        }
        if rows.iter().any(|row| row.len() != inputs) {
            return Err("its weight rows do not match the previous layer's size");
        }
        if biases.len() != rows.len() {
            return Err("its number of biases does not match its number of neurons");
        }

        Ok(Self {
            weights: DMatrix::from_fn(rows.len(), inputs, |neuron, input| rows[neuron][input]),
            biases: DVector::from_row_slice(biases),
            activation,
        })
    }

    /// Computes this layer's outputs into `outputs`, without allocating.
    ///
    /// `inputs` must hold exactly [`inputs()`](Self::inputs) values and `outputs`
    /// exactly [`size()`](Self::size) values.
    pub fn forward_into(&self, inputs: &[f64], outputs: &mut [f64]) {
        self.weighted_inputs_into(inputs, outputs);
        add_bias_and_activate(self.activation, outputs, self.biases.as_slice());
    }

    /// Computes this layer's outputs for every column of `inputs` into the matching
    /// column of `outputs`, without allocating.
    ///
    /// `inputs` must have [`inputs()`](Self::inputs) rows and `outputs`
    /// [`size()`](Self::size) rows, with one column per sample in both.
    pub fn forward_batch_into(&self, inputs: DMatrixView<'_, f64>, mut outputs: DMatrixViewMut<'_, f64>) {
        outputs.gemm(1.0, &self.weights, &inputs, 0.0);
        for mut column in outputs.column_iter_mut() {
            add_bias_and_activate(self.activation, column.as_mut_slice(), self.biases.as_slice());
        }
    }

    /// The number of neurons.
    pub fn size(&self) -> usize {
        self.weights.nrows()
    }

    /// The number of inputs each neuron receives.
    pub fn inputs(&self) -> usize {
        self.weights.ncols()
    }

    /// Computes this layer's weighted sums `W·x + b`, before the activation, into
    /// `pre_activations`, without allocating.
    ///
    /// `inputs` must hold exactly [`inputs()`](Self::inputs) values and
    /// `pre_activations` exactly [`size()`](Self::size) values. Applying
    /// [`activation()`](Self::activation) to each one gives what
    /// [`forward_into`](Self::forward_into) computes.
    pub fn pre_activations_into(&self, inputs: &[f64], pre_activations: &mut [f64]) {
        self.weighted_inputs_into(inputs, pre_activations);
        for (sum, bias) in pre_activations.iter_mut().zip(self.biases.iter()) {
            *sum += bias;
        }
    }

    /// Writes `W·x`, without the biases, into `sums`. This is the only place the
    /// single-sample product is computed. [`forward_into`](Self::forward_into) adds the
    /// biases in the same loop as the activation, and
    /// [`pre_activations_into`](Self::pre_activations_into) adds them on their own; both
    /// add each bias to the same product, so the trace stays bit-identical to
    /// `feed_forward`. It is forced inline so `forward_into`, the per-sample hot path,
    /// keeps the product and the activation loop in one function.
    #[inline(always)]
    fn weighted_inputs_into(&self, inputs: &[f64], sums: &mut [f64]) {
        let inputs = DVectorView::from_slice(inputs, self.inputs());
        let mut sums = DVectorViewMut::from_slice(sums, self.size());

        sums.gemv(1.0, &self.weights, &inputs, 0.0);
    }

    pub fn activation(&self) -> ActivationFunction {
        self.activation
    }

    pub fn set_activation(&mut self, activation: ActivationFunction) {
        self.activation = activation;
    }

    pub fn set_weight(&mut self, neuron: usize, input: usize, weight: f64) {
        self.weights[(neuron, input)] = weight;
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
        if biases.len() != self.biases.nrows() {
            panic!("Incompatible biases vector size: expected {} biases", self.biases.nrows());
        }
        self.biases.copy_from_slice(biases);
    }

    pub fn set_bias(&mut self, neuron: usize, bias: f64) {
        self.biases[neuron] = bias;
    }

    /// The weights as one row per neuron, each holding one weight per input.
    pub fn weight_rows(&self) -> Vec<Vec<f64>> {
        self.weights.row_iter().map(|row| row.iter().copied().collect()).collect()
    }

    pub fn biases(&self) -> &DVector<f64> {
        &self.biases
    }

    pub fn weights(&self) -> &DMatrix<f64> {
        &self.weights
    }

    /// How many values this layer contributes to the network's flat parameter view.
    pub fn parameter_count(&self) -> usize {
        Self::parameter_count_for(self.size(), self.inputs())
    }

    pub fn parameter_count_for(neurons: usize, inputs: usize) -> usize {
        neurons * (inputs + 1)
    }

    /// Appends this layer's parameters: the weights row by row, then the biases.
    pub fn extend_parameters(&self, parameters: &mut Vec<f64>) {
        for row in self.weights.row_iter() {
            parameters.extend(row.iter());
        }
        parameters.extend(self.biases.iter());
    }

    /// Reads this layer's parameters, in the order of [`Layer::extend_parameters`], from
    /// the front of `parameters`, and returns what is left.
    pub fn take_parameters<'a>(&mut self, parameters: &'a [f64]) -> &'a [f64] {
        let (neurons, inputs) = self.weights.shape();
        let (weights, rest) = parameters.split_at(neurons * inputs);
        let (biases, rest) = rest.split_at(neurons);

        // The slice is row-major and nalgebra stores columns, so copy element by element.
        for (neuron, row) in weights.chunks_exact(inputs).enumerate() {
            for (input, &weight) in row.iter().enumerate() {
                self.weights[(neuron, input)] = weight;
            }
        }
        self.biases.copy_from_slice(biases);

        rest
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer Size: {}", self.size())?;
        writeln!(f, "Activation Function: {:?}", self.activation)?;
        writeln!(f, "Weights:")?;
        for i in 0..self.weights.nrows() {
            for j in 0..self.weights.ncols() {
                write!(f, "{:0.2} ", self.weights[(i, j)])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Biases:")?;
        for bias in self.biases.iter() {
            writeln!(f, "{:0.2}", bias)?;
        }
        Ok(())
    }
}

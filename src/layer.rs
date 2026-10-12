use std::fmt;
use nalgebra::{DMatrix, DMatrixView, DMatrixViewMut, DVector, DVectorView, DVectorViewMut};
use rand::distr::Uniform;
use rand::Rng;

use crate::activation_functions::{add_bias_and_activate, ActivationFunction};

/// One layer of neurons: a weight per neuron and input, a bias per neuron, and the
/// activation applied to every neuron.
///
/// The weights are stored transposed, one row per input and one column per neuron.
/// nalgebra stores matrices column by column, so in memory that is every weight of
/// neuron 0, then of neuron 1, and so on: exactly the row-per-neuron order of the
/// flat parameter view and of the model file, which then copy whole slices. Every
/// method still takes and returns weights row per neuron.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// `inputs × neurons`, the transpose of the usual `W`.
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

        // Drawn input by input, each across every neuron, which is the order the
        // untransposed matrix was filled in, so a seed keeps giving the same network.
        let draws = std::iter::repeat_with(|| rng.sample(uniform));
        Self {
            weights: DMatrix::from_row_iterator(inputs, neurons, draws),
            biases: DVector::zeros(neurons),
            activation,
        }
    }

    /// A layer whose weights and biases are all zero.
    pub fn zeros(neurons: usize, inputs: usize, activation: ActivationFunction) -> Self {
        Self {
            weights: DMatrix::zeros(inputs, neurons),
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

        // The rows laid end to end are the transposed matrix's columns. Built as one
        // `Vec` of known size: nalgebra collects an iterator without a size hint by
        // growing a `Vec`, which made `read_model` 40% slower.
        let mut weights = Vec::with_capacity(rows.len() * inputs);
        for row in rows {
            weights.extend_from_slice(row);
        }

        Ok(Self {
            weights: DMatrix::from_vec(inputs, rows.len(), weights),
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
        add_bias_and_activate(self.activation, self.biases.as_slice(), [outputs]);
    }

    /// Computes this layer's outputs for every column of `inputs` into the matching
    /// column of `outputs`.
    ///
    /// `inputs` must have [`inputs()`](Self::inputs) rows and `outputs`
    /// [`size()`](Self::size) rows, with one column per sample in both.
    pub fn forward_batch_into(&self, inputs: DMatrixView<'_, f64>, mut outputs: DMatrixViewMut<'_, f64>) {
        // The one allocation: a copy of `W` in its usual layout. `gemm_tr` would skip
        // the copy but computes one dot product per output, which made the 64-128-64-10
        // batch benchmark 85% slower; `gemm` hands products this size to matrixmultiply.
        // A view of the storage with swapped strides would avoid the copy too, but
        // nalgebra's small-matrix `gemm` path reads past such a view's columns.
        outputs.gemm(1.0, &self.weights.transpose(), &inputs, 0.0);

        // Each column of the view is contiguous even though the view is not (it is the
        // top rows of a taller matrix), so it can be handed over as a slice. One call
        // covers every column, so the activation is matched once per layer.
        let columns = outputs.column_iter_mut().map(|column| column.data.into_slice_mut());
        add_bias_and_activate(self.activation, self.biases.as_slice(), columns);
    }

    /// The number of neurons.
    pub fn size(&self) -> usize {
        self.weights.ncols()
    }

    /// The number of inputs each neuron receives.
    pub fn inputs(&self) -> usize {
        self.weights.nrows()
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
    /// single-sample product is computed. The stored matrix is `Wᵀ`, so this is
    /// `gemv_tr`: one dot product over contiguous weights per neuron.
    /// [`forward_into`](Self::forward_into) adds the biases in the same loop as the
    /// activation, and [`pre_activations_into`](Self::pre_activations_into) adds them
    /// on their own; both add each bias to the same product, so the trace stays
    /// bit-identical to `feed_forward`. It is forced inline so `forward_into`, the per-sample hot path,
    /// keeps the product and the activation loop in one function.
    #[inline(always)]
    fn weighted_inputs_into(&self, inputs: &[f64], sums: &mut [f64]) {
        let inputs = DVectorView::from_slice(inputs, self.inputs());
        let mut sums = DVectorViewMut::from_slice(sums, self.size());

        sums.gemv_tr(1.0, &self.weights, &inputs, 0.0);
    }

    pub fn activation(&self) -> ActivationFunction {
        self.activation
    }

    pub fn set_activation(&mut self, activation: ActivationFunction) {
        self.activation = activation;
    }

    pub fn weight(&self, neuron: usize, input: usize) -> f64 {
        self.weights[(input, neuron)]
    }

    pub fn set_weight(&mut self, neuron: usize, input: usize, weight: f64) {
        self.weights[(input, neuron)] = weight;
    }

    /// Replaces every weight, given one row per neuron with one entry per input.
    pub fn set_weights<R: AsRef<[f64]>>(&mut self, rows: &[R]) {
        let (neurons, inputs) = (self.size(), self.inputs());
        if rows.len() != neurons || rows.iter().any(|row| row.as_ref().len() != inputs) {
            panic!("Incompatible weights matrix size: expected {neurons} rows of {inputs} weights");
        }
        // Each column holds one neuron's weights.
        for (mut column, row) in self.weights.column_iter_mut().zip(rows) {
            column.copy_from_slice(row.as_ref());
        }
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
        self.weights.column_iter().map(|column| column.iter().copied().collect()).collect()
    }

    pub fn biases(&self) -> &DVector<f64> {
        &self.biases
    }

    /// Every weight, row by row: all the weights of neuron 0, then of neuron 1, and so
    /// on, as in the flat parameter view.
    pub fn flat_weights(&self) -> &[f64] {
        self.weights.as_slice()
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
        parameters.extend_from_slice(self.flat_weights());
        parameters.extend_from_slice(self.biases.as_slice());
    }

    /// Calls `f` on each of this layer's parameters in place, in the order of
    /// [`Layer::extend_parameters`].
    pub fn for_each_parameter_mut(&mut self, f: &mut impl FnMut(&mut f64)) {
        self.weights.as_mut_slice().iter_mut().for_each(&mut *f);
        self.biases.as_mut_slice().iter_mut().for_each(f);
    }

    /// Reads this layer's parameters, in the order of [`Layer::extend_parameters`], from
    /// the front of `parameters`, and returns what is left.
    pub fn take_parameters<'a>(&mut self, parameters: &'a [f64]) -> &'a [f64] {
        let (weights, rest) = parameters.split_at(self.weights.len());
        let (biases, rest) = rest.split_at(self.biases.len());

        self.weights.as_mut_slice().copy_from_slice(weights);
        self.biases.as_mut_slice().copy_from_slice(biases);

        rest
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer Size: {}", self.size())?;
        writeln!(f, "Activation Function: {:?}", self.activation)?;
        writeln!(f, "Weights:")?;
        for column in self.weights.column_iter() {
            for weight in column.iter() {
                write!(f, "{:0.2} ", weight)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: [[f64; 2]; 3] = [[0.5, -0.25], [-1.0, 2.0], [0.0, 0.75]];
    const BIASES: [f64; 3] = [0.1, -0.3, 0.0];
    const SAMPLES: [[f64; 2]; 4] = [[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0], [3.0, -2.0]];

    fn layer(activation: ActivationFunction) -> Layer {
        let rows: Vec<Vec<f64>> = ROWS.iter().map(|row| row.to_vec()).collect();
        Layer::from_parts(&rows, &BIASES, activation, 2).expect("the parts form a layer")
    }

    /// What the layer outputs for one sample, computed the slow way through `apply`.
    fn expected(activation: ActivationFunction, sample: [f64; 2]) -> Vec<f64> {
        ROWS.iter()
            .zip(BIASES)
            .map(|(row, bias)| activation.apply(row[0] * sample[0] + row[1] * sample[1] + bias))
            .collect()
    }

    fn assert_all_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "expected {expected:?}, got {actual:?}");
        }
    }

    #[test]
    fn forward_into_applies_every_activation_function_to_each_neuron() {
        for activation in ActivationFunction::ALL {
            let layer = layer(activation);

            for sample in SAMPLES {
                let mut outputs = [f64::NAN; 3];
                layer.forward_into(&sample, &mut outputs);

                assert_all_close(&outputs, &expected(activation, sample));
            }
        }
    }

    /// The outputs are the top rows of a taller matrix, as in `feed_forward_batch`, so
    /// the columns are not contiguous with each other and the rows below must not be
    /// touched.
    #[test]
    fn forward_batch_into_applies_every_activation_function_to_each_column() {
        for activation in ActivationFunction::ALL {
            let layer = layer(activation);
            let inputs = DMatrix::from_fn(2, SAMPLES.len(), |input, sample| SAMPLES[sample][input]);
            let mut outputs = DMatrix::from_element(5, SAMPLES.len(), f64::NAN);

            layer.forward_batch_into(inputs.rows(0, 2), outputs.rows_mut(0, 3));

            for (column, sample) in outputs.column_iter().zip(SAMPLES) {
                let column: Vec<f64> = column.iter().copied().collect();
                assert_all_close(&column[..3], &expected(activation, sample));
                assert!(column[3..].iter().all(|value| value.is_nan()), "{activation:?} wrote past its rows");
            }
        }
    }
}

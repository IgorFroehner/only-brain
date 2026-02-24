
use polars::prelude::*;
use polars::lazy::dsl::{col, lit};
use itertools::izip;

use only_brain::{Perceptron, ActivationFunction, bvector};

fn main() -> PolarsResult<()> {

    // 1. Load the Iris dataset and simplify the problem to binary classification
    //    so that a perceptron can be better used.

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/datasets/iris.csv".into()))?
        .finish()?;

    // Keep only two classes: Iris-setosa and Iris-versicolor
    let df = df
        .lazy()
        .filter(
            col("Species")
                .eq(lit("Iris-setosa"))
                .or(col("Species").eq(lit("Iris-versicolor"))),
        )
        .collect()?;

    let remove_cols = ["Id", "PetalLengthCm", "PetalWidthCm"];
    let df = df.drop_many(remove_cols);

    // Map species strings to f64 (0.0 / 1.0)
    let species_f64 = df
        .column("Species")?        // &Series
        .str()?                    // &StringChunked
        .into_iter()               // Iterator<Option<&str>>
        .map(|opt_name| match opt_name {
            Some("Iris-setosa") => Some(0.0),
            Some("Iris-versicolor") => Some(1.0),
            _ => None,
        })
        .collect::<Float64Chunked>() // Collect into Float64 column
        .into_series();

    // Replace original Species column with numeric version
    let mut df = df.clone();
    df.replace("Species", species_f64)?;

    println!("Filtered DataFrame:\n{df}");

    // Extract typed columns
    let sepal_length = df.column("SepalLengthCm")?.f64()?;
    let sepal_width = df.column("SepalWidthCm")?.f64()?;
    let species = df.column("Species")?.f64()?;

    let eta = 0.1;

    let mut perceptron = Perceptron::<2>::new(ActivationFunction::BinaryStep);

    let mut weights = bvector![1.0, 1.0];
    let mut bias = 0.0;

    perceptron.set_weights(weights.clone());
    perceptron.set_bias(bias);

    let vectors = izip!(
        sepal_length.into_no_null_iter(),
        sepal_width.into_no_null_iter(),
        species.into_no_null_iter()
    );

    // gradient descent learning
    for (sl, sw, sp) in vectors {
        let output = perceptron.feed_forward(&bvector![sl, sw]);
        if output != sp {
            weights = bvector![
                weights.get(0) + eta * sp * sl,
                weights.get(1) + eta * sp * sw,
            ];
            bias = bias + eta * sp;
            perceptron.set_weights(weights.clone());
            perceptron.set_bias(bias);
        }
        println!("Sepal length: {sl}, Sepal width: {sw}, Species: {sp}, Output: {output}");
    }

    println!("\nFinal Perceptron: {}", perceptron);

    Ok(())
}

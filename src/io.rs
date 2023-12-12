use std::fs::File;
use std::io::Write;
use crate::NeuralNetwork;

pub fn dump_model(model: &NeuralNetwork, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(model)?;

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;

    Ok(())
}

pub fn load_model(path: &str) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let model = bincode::deserialize_from(file)?;

    Ok(model)
}

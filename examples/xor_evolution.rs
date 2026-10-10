//! Evolves a network that computes XOR, using only the flat parameter view.
//!
//! The genetic algorithm here never sees a layer: each individual is a plain
//! `Vec<f64>` genome of `parameter_count_for(HIDDEN)` genes, and the fitness function
//! turns it into a network with `from_parameters`. That is the whole bridge between
//! this library and a genetic algorithm such as the `evolutionary` crate.
//!
//! It uses the settings that suit evolving network weights: tournament selection,
//! Gaussian mutation and elitism, and no crossover, since crossing two networks'
//! weights tends to break both parents.
//!
//! Run with `cargo run --example xor_evolution`.

use only_brain::{dump_model, load_model, ActivationFunction, NeuralNetwork};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

type Pilot = NeuralNetwork<2, 1>;

const HIDDEN: &[usize] = &[3];
const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

const SEED: u64 = 2024;
const POPULATION: usize = 150;
const ELITES: usize = 2;
const TOURNAMENT: usize = 3;
const MUTATION_RATE: f64 = 0.2;
const MUTATION_STRENGTH: f64 = 0.4;
const MAX_GENERATIONS: usize = 1000;
/// Every case must be within this distance of its target.
const TOLERANCE: f64 = 0.1;

/// Builds the network a genome stands for.
fn network(genome: &[f64]) -> Pilot {
    let mut nn = Pilot::from_parameters(HIDDEN, genome);
    nn.set_activation_function(ActivationFunction::Tanh);
    nn.set_output_activation(ActivationFunction::Sigmoid);
    nn
}

/// Higher is better: the negated mean squared error over the four cases.
fn fitness(genome: &[f64]) -> f64 {
    let nn = network(genome);
    let squared_error: f64 = CASES
        .iter()
        .map(|(inputs, target)| (nn.feed_forward(inputs)[0] - target).powi(2))
        .sum();

    -squared_error / CASES.len() as f64
}

fn solves_xor(nn: &Pilot) -> bool {
    CASES
        .iter()
        .all(|(inputs, target)| (nn.feed_forward(inputs)[0] - target).abs() < TOLERANCE)
}

/// Picks the fittest of a few random individuals.
fn tournament<'a>(scored: &'a [(f64, Vec<f64>)], rng: &mut impl Rng) -> &'a [f64] {
    let (_, genome) = (0..TOURNAMENT)
        .map(|_| scored.choose(rng).expect("the population is not empty"))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .expect("the tournament is not empty");
    genome
}

fn mutate(genome: &mut [f64], rng: &mut impl Rng) {
    let noise = Normal::new(0.0, MUTATION_STRENGTH).expect("the deviation is positive");
    for gene in genome {
        if rng.random_bool(MUTATION_RATE) {
            *gene += noise.sample(rng);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let genes = Pilot::parameter_count_for(HIDDEN);
    let initial = Normal::new(0.0, 1.0)?;

    println!("Evolving a 2 -> {HIDDEN:?} -> 1 network: {genes} genes, {POPULATION} individuals");

    let mut population: Vec<Vec<f64>> = (0..POPULATION)
        .map(|_| (0..genes).map(|_| initial.sample(&mut rng)).collect())
        .collect();

    let mut champion = Vec::new();
    for generation in 0..MAX_GENERATIONS {
        let mut scored: Vec<(f64, Vec<f64>)> = population
            .into_iter()
            .map(|genome| (fitness(&genome), genome))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        let (best_fitness, best) = &scored[0];
        champion.clone_from(best);
        if generation % 25 == 0 || solves_xor(&network(best)) {
            println!("generation {generation:4}: best mean squared error {:.5}", -best_fitness);
        }
        if solves_xor(&network(best)) {
            println!("Solved XOR in generation {generation}");
            break;
        }

        // Elites pass unchanged; everyone else is a mutated tournament winner.
        population = scored[..ELITES].iter().map(|(_, genome)| genome.clone()).collect();
        while population.len() < POPULATION {
            let mut child = tournament(&scored, &mut rng).to_vec();
            mutate(&mut child, &mut rng);
            population.push(child);
        }
    }

    let champion = network(&champion);
    println!();
    for (inputs, target) in CASES {
        let [output] = champion.feed_forward(&inputs);
        println!("{inputs:?} -> {output:.3} (target {target})");
    }

    // Save the champion and load it back, as a training run would to replay it later.
    let path = std::env::temp_dir().join("only-brain-xor-champion.bin");
    dump_model(&champion, &path)?;
    let replayed: Pilot = load_model(&path)?;
    assert_eq!(replayed, champion);
    println!("\nChampion saved to {} and reloaded", path.display());

    Ok(())
}

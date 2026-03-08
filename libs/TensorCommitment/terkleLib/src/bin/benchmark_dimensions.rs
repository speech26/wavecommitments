//! Benchmark script to compare tree construction times across different dimensions.
//! Saves results to a JSONL file for plotting.
//!
//! Usage: cargo run --release --bin benchmark_dimensions [output_file] [num_seeds] [iterations_per_seed]

use terkle::{MultiverkleConfig, MultiverkleTree};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::Instant;

type Scalar = <Bn254 as Pairing>::ScalarField;

#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkResult {
    seed: u64,
    config_name: String,
    axis_arity: Vec<usize>,
    axis_count: usize,
    degree_bound: usize,
    srs_size: usize,
    data_size: usize,
    depth: usize,
    build_time_seconds: f64,
    proof_time_seconds: f64,
    verify_time_seconds: f64,
}

fn run_single_benchmark(
    config_name: &str,
    axis_arity: Vec<usize>,
    data_size: usize,
    depth: usize,
    seed: u64,
) -> BenchmarkResult {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let config = MultiverkleConfig {
        axis_arity: axis_arity.clone(),
        depth,
    };

    let degree_bound = *axis_arity.iter().max().unwrap();
    let axis_count = axis_arity.len();
    let srs_size = axis_count * degree_bound;

    // Generate test data (with seed-based offset for variety)
    let data: Vec<Scalar> = (0..data_size)
        .map(|i| Scalar::from(i as u64 + seed * 1000))
        .collect();

    // Measure tree construction
    let start = Instant::now();
    let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone())
        .unwrap();
    let build_time = start.elapsed().as_nanos() as f64 / 1_000_000_000.0; // Convert to seconds

    // Measure proof generation and verification (sample one index)
    let index = (seed as usize * 17) % data_size; // Deterministic but varied index
    let path = tree.path_from_linear_index(index).unwrap();
    
    // Proof generation time
    let start = Instant::now();
    let proof = tree.open(&path).unwrap();
    let proof_time = start.elapsed().as_nanos() as f64 / 1_000_000_000.0; // Convert to seconds

    // Verification time (stateless - only needs config, vk, root, proof)
    let expected_value = data[index];
    let root_commitment = tree.root_commitment();
    let vk = tree.verification_key();
    
    let start = Instant::now();
    let _is_valid = MultiverkleTree::verify_membership(
        &config,
        index,
        &expected_value,
        vk,
        root_commitment,
        &proof,
    ).unwrap();
    let verify_time = start.elapsed().as_nanos() as f64 / 1_000_000_000.0; // Convert to seconds

    BenchmarkResult {
        seed,
        config_name: config_name.to_string(),
        axis_arity,
        axis_count,
        degree_bound,
        srs_size,
        data_size,
        depth,
        build_time_seconds: build_time,
        proof_time_seconds: proof_time,
        verify_time_seconds: verify_time,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let output_file = args.get(1).map(|s| s.as_str()).unwrap_or("dimension_benchmark.jsonl");
    let num_seeds: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let iterations_per_seed: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Configurations: all have fanout=64
    let configs = vec![
        ("2D [8,8]", vec![8, 8]),
        ("3D [4,4,4]", vec![4, 4, 4]),
        ("6D [2,2,2,2,2,2]", vec![2, 2, 2, 2, 2, 2]),
    ];

    // Data sizes: 64, 64², 64³, 64⁴
    let data_sizes = vec![
        (64, 1),
        (4096, 2),
        (262144, 3),
        (16777216, 4),
    ];

    eprintln!("Starting dimension comparison benchmark...");
    eprintln!("Output file: {}", output_file);
    eprintln!("Number of seeds: {}", num_seeds);
    eprintln!("Iterations per seed: {}", iterations_per_seed);
    eprintln!("Total runs per configuration: {}", num_seeds * iterations_per_seed);
    eprintln!();

    // Open file in append mode to allow resuming
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_file)?;

    let mut total_runs = 0;
    
    for (data_size, depth) in &data_sizes {
        eprintln!("Testing with {} data points (depth={})", 
                 if *data_size >= 1_000_000 {
                     format!("{:.1}M", *data_size as f64 / 1_000_000.0)
                 } else if *data_size >= 1_000 {
                     format!("{:.1}K", *data_size as f64 / 1_000.0)
                 } else {
                     data_size.to_string()
                 }, depth);
        
        for (config_name, axis_arity) in &configs {
            eprint!("  Running {}... ", config_name);
            io::stderr().flush()?;
            
            let mut runs_this_config = 0;
            
            // Run with different seeds
            for seed in 1..=num_seeds {
                // Run multiple iterations per seed
                for iter in 0..iterations_per_seed {
                    let effective_seed = (seed * 1000 + iter) as u64; // Unique seed per run
                    
                    let result = run_single_benchmark(
                        config_name,
                        axis_arity.clone(),
                        *data_size,
                        *depth,
                        effective_seed,
                    );
                    
                    // Write individual result to JSONL file
                    let json_line = serde_json::to_string(&result)?;
                    writeln!(file, "{}", json_line)?;
                    file.flush()?;
                    
                    runs_this_config += 1;
                    total_runs += 1;
                }
            }
            
            eprintln!("✓ ({} runs completed)", runs_this_config);
        }
        eprintln!();
    }

    eprintln!("✓ Benchmark complete! {} total runs saved to {}", total_runs, output_file);
    Ok(())
}


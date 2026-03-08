//! Detailed benchmark that measures individual operations separately
//! to identify where the theoretical advantage of higher dimensions should manifest.
//!
//! Measures:
//! - IFFT time per node
//! - Padding time per node
//! - PST commit time per node
//! - Total tree construction time
//!
//! This helps identify bottlenecks and verify that optimizations are working.

use terkle::{MultiverkleConfig, MultiverkleTree};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::{PrimeField, Zero};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use tensor_commitment_lib::PST;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::time::Instant;

type Scalar = <Bn254 as Pairing>::ScalarField;
type G1 = <Bn254 as Pairing>::G1;
type G1Affine = <Bn254 as Pairing>::G1Affine;

#[derive(Serialize, Deserialize, Debug)]
struct DetailedResult {
    seed: u64,
    config_name: String,
    axis_arity: Vec<usize>,
    axis_count: usize,
    degree_bound: usize,
    data_size: usize,
    depth: usize,
    
    // Per-node operation times (averaged across all nodes in tree)
    avg_ifft_time_us: f64,
    avg_padding_time_us: f64,
    avg_commit_time_us: f64,
    avg_digest_time_us: f64,
    
    // Tree-level timings
    total_build_time_ms: f64,
    total_nodes: usize,
    
    // Proof and verification
    proof_time_us: f64,
    verify_time_us: f64,
}

fn benchmark_with_profiling(
    config_name: &str,
    axis_arity: Vec<usize>,
    data_size: usize,
    depth: usize,
    seed: u64,
) -> DetailedResult {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let config = MultiverkleConfig {
        axis_arity: axis_arity.clone(),
        depth,
    };

    let degree_bound = *axis_arity.iter().max().unwrap();
    let axis_count = axis_arity.len();
    
    // Generate test data
    let data: Vec<Scalar> = (0..data_size)
        .map(|i| Scalar::from(i as u64 + seed * 1000))
        .collect();

    // Build domains
    let domains: Vec<GeneralEvaluationDomain<Scalar>> = axis_arity
        .iter()
        .map(|&n| GeneralEvaluationDomain::new(n).expect("FFT domain"))
        .collect();
    
    // Setup PST
    let (ck, _vk) = PST::<Bn254>::setup(&mut rng, axis_count, degree_bound);
    
    // Count nodes for averaging
    let fanout = axis_arity.iter().product::<usize>();
    let mut total_nodes = 0;
    let mut level_count = 1;
    for _ in 0..=depth {
        total_nodes += level_count;
        level_count *= fanout;
    }
    
    // Measure per-node operations on a sample
    let sample_size = 64; // Sample one node's operations
    let sample_data: Vec<Scalar> = data[..sample_size.min(data_size)].to_vec();
    
    // Benchmark IFFT
    let mut ifft_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _coeffs = tensor_ifft_sample(&sample_data, &axis_arity, &domains);
        ifft_times.push(start.elapsed());
    }
    let avg_ifft_us = ifft_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / ifft_times.len() as f64 / 1_000.0;
    
    // Benchmark padding
    let tensor_coeffs = tensor_ifft_sample(&sample_data, &axis_arity, &domains);
    let mut pad_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _padded = pad_coeffs_sample(&tensor_coeffs, &axis_arity, degree_bound);
        pad_times.push(start.elapsed());
    }
    let avg_pad_us = pad_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / pad_times.len() as f64 / 1_000.0;
    
    // Benchmark PST commit
    let padded = pad_coeffs_sample(&tensor_coeffs, &axis_arity, degree_bound);
    let mut commit_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _commit = PST::<Bn254>::commit(&ck, &padded);
        commit_times.push(start.elapsed());
    }
    let avg_commit_us = commit_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / commit_times.len() as f64 / 1_000.0;
    
    // Benchmark digest
    let commit = PST::<Bn254>::commit(&ck, &padded);
    let mut digest_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _digest = digest_commitment_sample(&commit);
        digest_times.push(start.elapsed());
    }
    let avg_digest_us = digest_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / digest_times.len() as f64 / 1_000.0;
    
    // Now measure full tree construction
    let start = Instant::now();
    let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();
    let total_build_ms = start.elapsed().as_nanos() as f64 / 1_000_000.0;
    
    // Measure proof and verification
    let index = (seed as usize * 17) % data_size;
    let path = tree.path_from_linear_index(index).unwrap();
    
    let start = Instant::now();
    let proof = tree.open(&path).unwrap();
    let proof_time_us = start.elapsed().as_nanos() as f64 / 1_000.0;
    
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
    let verify_time_us = start.elapsed().as_nanos() as f64 / 1_000.0;
    
    DetailedResult {
        seed,
        config_name: config_name.to_string(),
        axis_arity,
        axis_count,
        degree_bound,
        data_size,
        depth,
        avg_ifft_time_us: avg_ifft_us,
        avg_padding_time_us: avg_pad_us,
        avg_commit_time_us: avg_commit_us,
        avg_digest_time_us: avg_digest_us,
        total_build_time_ms: total_build_ms,
        total_nodes,
        proof_time_us,
        verify_time_us,
    }
}

// Helper functions (simplified versions from lib.rs)
fn tensor_ifft_sample(
    values: &[Scalar],
    axis_arity: &[usize],
    domains: &[GeneralEvaluationDomain<Scalar>],
) -> Vec<Scalar> {
    let mut data = values.to_vec();
    let axis_count = axis_arity.len();
    let mut stride = 1usize;

    for axis in (0..axis_count).rev() {
        let axis_size = axis_arity[axis];
        let domain = &domains[axis];
        let chunk_len = stride * axis_size;

        for chunk_start in (0..data.len()).step_by(chunk_len) {
            let chunk = &mut data[chunk_start..chunk_start + chunk_len];
            for offset in 0..stride {
                let mut evals = Vec::with_capacity(axis_size);
                for i in 0..axis_size {
                    let idx = offset + i * stride;
                    evals.push(chunk[idx]);
                }
                domain.ifft_in_place(&mut evals);
                for (i, coeff) in evals.into_iter().enumerate() {
                    let idx = offset + i * stride;
                    chunk[idx] = coeff;
                }
            }
        }
        stride *= axis_size;
    }
    data
}

fn pad_coeffs_sample(coeffs: &[Scalar], axis_arity: &[usize], degree_bound: usize) -> Vec<Scalar> {
    let axis_count = axis_arity.len();
    let padded_len = degree_bound.pow(axis_count as u32);
    let mut padded = vec![Scalar::zero(); padded_len];
    
    fn linear_to_coords(mut idx: usize, axis_arity: &[usize]) -> Vec<usize> {
        let mut coords = vec![0usize; axis_arity.len()];
        for axis in (0..axis_arity.len()).rev() {
            let base = axis_arity[axis];
            coords[axis] = idx % base;
            idx /= base;
        }
        coords
    }
    
    fn lex_index(coords: &[usize], degree_bound: usize) -> usize {
        coords.iter().fold(0usize, |acc, &coord| {
            acc * degree_bound + coord
        })
    }
    
    for (offset, coeff) in coeffs.iter().enumerate() {
        let coords = linear_to_coords(offset, axis_arity);
        let idx = lex_index(&coords, degree_bound);
        padded[idx] = *coeff;
    }
    padded
}

fn digest_commitment_sample(point: &G1) -> Scalar {
    use ark_ec::CurveGroup;
    use ark_serialize::CanonicalSerialize;
    let affine = (*point).into_affine();
    let mut bytes = Vec::new();
    affine.serialize_compressed(&mut bytes).unwrap();
    Scalar::from_le_bytes_mod_order(&bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let output_file = args.get(1).map(|s| s.as_str()).unwrap_or("detailed_benchmark.jsonl");
    let num_seeds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iterations_per_seed: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let configs = vec![
        ("2D [8,8]", vec![8, 8]),
        ("3D [4,4,4]", vec![4, 4, 4]),
        ("6D [2,2,2,2,2,2]", vec![2, 2, 2, 2, 2, 2]),
    ];

    let data_sizes = vec![
        (64, 1),
        (4096, 2),
        (262144, 3),
        (16777216, 4),
        (1073741824, 5),
    ];

    eprintln!("Starting detailed benchmark with operation-level profiling...");
    eprintln!("Output file: {}", output_file);
    eprintln!("Seeds: {}, Iterations per seed: {}", num_seeds, iterations_per_seed);
    eprintln!();

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_file)?;

    let mut total_runs = 0;
    
    for (data_size, depth) in &data_sizes {
        eprintln!("Testing {} data points (depth={})", 
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
            
            for seed in 1..=num_seeds {
                for iter in 0..iterations_per_seed {
                    let effective_seed = (seed * 1000 + iter) as u64;
                    let result = benchmark_with_profiling(
                        config_name,
                        axis_arity.clone(),
                        *data_size,
                        *depth,
                        effective_seed,
                    );
                    
                    let json_line = serde_json::to_string(&result)?;
                    writeln!(file, "{}", json_line)?;
                    file.flush()?;
                    
                    total_runs += 1;
                }
            }
            
            eprintln!("✓");
        }
        eprintln!();
    }

    eprintln!("✓ Detailed benchmark complete! {} runs saved to {}", total_runs, output_file);
    Ok(())
}


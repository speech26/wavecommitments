// pyo3_bindings.rs - Python bindings for the PST library
// This file should be added to src/lib.rs or as a separate module

use ark_bn254::{Bn254, Fr};
use ark_ff::{PrimeField, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyList, PyModule};
use pyo3::{IntoPy, PyObject, PyResult, Python};
use std::sync::OnceLock;
use crate::PST;

fn ark_bigint_to_bigint(value: <Fr as PrimeField>::BigInt) -> BigInt {
    let mut bytes = Vec::with_capacity(value.0.len() * 8);
    for limb in value.0 {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    BigInt::from(BigUint::from_bytes_le(&bytes))
}

fn fr_modulus() -> &'static BigInt {
    static MODULUS: OnceLock<BigInt> = OnceLock::new();
    MODULUS.get_or_init(|| ark_bigint_to_bigint(Fr::MODULUS))
}

fn py_any_to_bigint(value: &PyAny) -> PyResult<BigInt> {
    value.extract::<BigInt>()
}

fn py_any_to_fr(value: &PyAny) -> PyResult<Fr> {
    let modulus = fr_modulus();
    let integer = py_any_to_bigint(value)?;
    let reduced = integer.mod_floor(modulus);
    let magnitude = reduced
        .to_biguint()
        .ok_or_else(|| PyValueError::new_err("Expected a finite integer value"))?;
    let bytes = magnitude.to_bytes_le();
    Ok(Fr::from_le_bytes_mod_order(&bytes))
}

fn py_list_to_fr_vec(list: &PyList) -> PyResult<Vec<Fr>> {
    list.iter().map(|item| py_any_to_fr(item)).collect()
}

fn fr_to_bigint(value: Fr) -> BigInt {
    ark_bigint_to_bigint(value.into_bigint())
}

/// Python wrapper for PST functionality
#[pyclass]
pub struct PSTWrapper {
    commitment_key: Vec<ark_bn254::G1Affine>,
    verification_key: Vec<ark_bn254::G2Affine>,
    degree_bound: usize,
    num_variables: usize,
}

#[pymethods]
impl PSTWrapper {
    #[new]
    fn new(num_variables: usize, degree_bound: usize) -> Self {
        let rng = &mut test_rng();
        let (ck, vk) = PST::<Bn254>::setup(rng, num_variables, degree_bound);
        
        PSTWrapper {
            commitment_key: ck,
            verification_key: vk,
            degree_bound,
            num_variables,
        }
    }
    
    /// Commit to polynomial coefficients
    fn commit(&self, coefficients: &PyList) -> PyResult<String> {
        let coeffs = py_list_to_fr_vec(coefficients)?;
        
        let commitment = PST::<Bn254>::commit(&self.commitment_key, &coeffs);
        
        // Serialize commitment to hex string for Python
        let mut bytes = Vec::new();
        commitment.serialize_compressed(&mut bytes).unwrap();
        Ok(hex::encode(bytes))
    }
    
    /// Prove polynomial evaluation at given point
    fn prove(&self, coefficients: &PyList, evaluation_point: &PyList, _claimed_evaluation: &PyAny) -> PyResult<Vec<String>> {
        let coeffs = py_list_to_fr_vec(coefficients)?;
        let point = py_list_to_fr_vec(evaluation_point)?;
        
        let proof = PST::<Bn254>::prove(self.degree_bound, &self.commitment_key, &point, &coeffs);
        
        // Serialize proof elements to hex strings
        let mut proof_hex = Vec::new();
        for pi in proof {
            let mut bytes = Vec::new();
            pi.serialize_compressed(&mut bytes).unwrap();
            proof_hex.push(hex::encode(bytes));
        }
        
        Ok(proof_hex)
    }
    
    /// Verify polynomial evaluation proof
    fn verify(&self, commitment_hex: &str, evaluation_point: &PyList, claimed_evaluation: &PyAny, proof_hex: &PyList) -> PyResult<bool> {
        // Deserialize commitment
        let commitment_bytes = hex::decode(commitment_hex).unwrap();
        let commitment = ark_bn254::G1::deserialize_compressed(&commitment_bytes[..]).unwrap();
        
        // Deserialize proof
        let mut proof = Vec::new();
        for item in proof_hex.iter() {
            let hex_str: &str = item.extract()?;
            let bytes = hex::decode(hex_str).unwrap();
            let pi = ark_bn254::G1::deserialize_compressed(&bytes[..]).unwrap();
            proof.push(pi);
        }
        
        // Parse evaluation point
        let point = py_list_to_fr_vec(evaluation_point)?;
        
        let claimed_eval = py_any_to_fr(claimed_evaluation)?;
        
        Ok(PST::<Bn254>::verify(&self.verification_key, &point, commitment, claimed_eval, &proof))
    }
    
    /// Get polynomial evaluation at a point (for testing)
    fn evaluate_polynomial(&self, py: Python, coefficients: &PyList, evaluation_point: &PyList) -> PyResult<PyObject> {
        let coeffs = py_list_to_fr_vec(coefficients)?;
        let point = py_list_to_fr_vec(evaluation_point)?;
        
        // Compute polynomial evaluation
        let mut ts_pows = vec![Fr::one()];
        for t in &point {
            let t_pows = crate::pows(*t, self.degree_bound);
            ts_pows = ts_pows
                .into_iter()
                .flat_map(|p| t_pows.iter().map(move |t| p * t))
                .collect();
        }
        
        let evaluation = coeffs
            .iter()
            .zip(&ts_pows)
            .map(|(fi, ti)| *fi * ti)
            .sum::<Fr>();
        
    // Convert back to integer (simplified)
    let eval_bigint = fr_to_bigint(evaluation);
    Ok(eval_bigint.into_py(py))
    }
}

/// Python module definition
#[pymodule]
fn tensor_commitment_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PSTWrapper>()?;
    Ok(())
}

#![allow(non_local_definitions)]

use std::io::Cursor;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{
    linear_index_to_path, scalar_to_decimal, G2Affine, MultiIndex, MultiverkleProof,
    MultiverkleTree, Scalar, TreeConfig, TreeError, G1,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::thread_rng;

#[pyclass(name = "MultiverkleTree")]
pub struct PyMultiverkleTree {
    inner: MultiverkleTree,
    axis_arity: Vec<usize>,
    depth: usize,
}

#[pyclass(name = "ProofBundle")]
pub struct PyProofBundle {
    proof: MultiverkleProof,
    linear_index: usize,
    path: Vec<Vec<usize>>,
}

#[pymethods]
impl PyMultiverkleTree {
    #[new]
    pub fn new(axis_arity: Vec<usize>, depth: usize, data: Vec<u64>) -> PyResult<Self> {
        if axis_arity.is_empty() {
            return Err(PyValueError::new_err(
                "axis_arity must contain at least one dimension",
            ));
        }
        if depth == 0 {
            return Err(PyValueError::new_err("depth must be >= 1"));
        }
        let config = TreeConfig {
            axis_arity: axis_arity.clone(),
            depth,
        };
        let expected = config.expected_value_count();
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "data length {} does not match expected {} for the provided config",
                data.len(),
                expected
            )));
        }

        let scalars = data.into_iter().map(Scalar::from).collect::<Vec<_>>();
        let mut rng = thread_rng();
        let inner = MultiverkleTree::from_data(&mut rng, config.clone(), scalars)
            .map_err(tree_error_to_py)?;

        Ok(Self {
            inner,
            axis_arity,
            depth,
        })
    }

    pub fn root_bytes(&self) -> PyResult<Vec<u8>> {
        serialize_g1(&self.inner.root_commitment())
    }

    pub fn verifier_key_bytes(&self) -> PyResult<Vec<u8>> {
        serialize_g2_vec(self.inner.verification_key())
    }

    pub fn axis_arity(&self) -> PyResult<Vec<usize>> {
        Ok(self.axis_arity.clone())
    }

    pub fn depth(&self) -> PyResult<usize> {
        Ok(self.depth)
    }

    pub fn open_index(&self, index: usize) -> PyResult<PyProofBundle> {
        let proof = self
            .inner
            .open_linear_index(index)
            .map_err(tree_error_to_py)?;
        let path = self
            .inner
            .path_from_linear_index(index)
            .map_err(tree_error_to_py)?;
        Ok(PyProofBundle::new(index, path, proof))
    }
}

impl PyProofBundle {
    fn new(index: usize, path: Vec<MultiIndex>, proof: MultiverkleProof) -> Self {
        let coords = path.into_iter().map(|mi| mi.coords).collect::<Vec<_>>();
        Self {
            proof,
            linear_index: index,
            path: coords,
        }
    }
}

#[pymethods]
impl PyProofBundle {
    pub fn proof_bytes(&self) -> PyResult<Vec<u8>> {
        serialize_proof(&self.proof)
    }

    pub fn proof_size(&self) -> PyResult<usize> {
        Ok(self.proof_bytes()?.len())
    }

    pub fn leaf_value(&self) -> PyResult<String> {
        Ok(scalar_to_decimal(&self.proof.leaf_value))
    }

    pub fn path(&self) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.path.clone())
    }

    pub fn linear_index(&self) -> PyResult<usize> {
        Ok(self.linear_index)
    }
}

#[pyfunction]
pub fn verify_serialized_proof(
    vk_bytes: Vec<u8>,
    root_bytes: Vec<u8>,
    axis_arity: Vec<usize>,
    depth: usize,
    index: usize,
    value: u128,
    proof_bytes: Vec<u8>,
) -> PyResult<bool> {
    let vk = deserialize_g2_vec(&vk_bytes)?;
    let root = deserialize_g1(&root_bytes)?;
    let proof = MultiverkleProof::deserialize_compressed(&mut Cursor::new(proof_bytes))
        .map_err(serialization_error)?;
    let config = TreeConfig { axis_arity, depth };
    let claimed_value = Scalar::from(value);
    MultiverkleTree::verify_membership(&config, index, &claimed_value, &vk, root, &proof)
        .map_err(tree_error_to_py)
}

#[pyfunction]
pub fn expected_value_count(axis_arity: Vec<usize>, depth: usize) -> PyResult<usize> {
    let config = TreeConfig { axis_arity, depth };
    Ok(config.expected_value_count())
}

#[pyfunction(name = "linear_index_to_path")]
pub fn linear_index_path(
    index: usize,
    axis_arity: Vec<usize>,
    depth: usize,
) -> PyResult<Vec<Vec<usize>>> {
    let path = linear_index_to_path(index, &axis_arity, depth).map_err(tree_error_to_py)?;
    Ok(path.into_iter().map(|mi| mi.coords).collect())
}

#[pymodule]
pub fn terkle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMultiverkleTree>()?;
    m.add_class::<PyProofBundle>()?;
    m.add_function(wrap_pyfunction!(verify_serialized_proof, m)?)?;
    m.add_function(wrap_pyfunction!(expected_value_count, m)?)?;
    m.add_function(wrap_pyfunction!(linear_index_path, m)?)?;
    Ok(())
}

fn serialize_g1(point: &G1) -> PyResult<Vec<u8>> {
    let mut bytes = Vec::new();
    point
        .serialize_compressed(&mut bytes)
        .map_err(serialization_error)?;
    Ok(bytes)
}

fn deserialize_g1(bytes: &[u8]) -> PyResult<G1> {
    G1::deserialize_compressed(&mut Cursor::new(bytes)).map_err(serialization_error)
}

fn serialize_g2_vec(points: &[G2Affine]) -> PyResult<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(points.len() as u32).to_le_bytes());
    for point in points {
        point
            .serialize_compressed(&mut bytes)
            .map_err(serialization_error)?;
    }
    Ok(bytes)
}

fn deserialize_g2_vec(bytes: &[u8]) -> PyResult<Vec<G2Affine>> {
    if bytes.len() < 4 {
        return Err(PyValueError::new_err("invalid verifier key payload"));
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut cursor = Cursor::new(&bytes[4..]);
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let point = G2Affine::deserialize_compressed(&mut cursor).map_err(serialization_error)?;
        result.push(point);
    }
    Ok(result)
}

fn serialize_proof(proof: &MultiverkleProof) -> PyResult<Vec<u8>> {
    let mut bytes = Vec::new();
    proof
        .serialize_compressed(&mut bytes)
        .map_err(serialization_error)?;
    Ok(bytes)
}

fn serialization_error(err: ark_serialize::SerializationError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn tree_error_to_py(err: TreeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

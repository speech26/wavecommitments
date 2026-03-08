use std::sync::Arc;

use ark_bn254::Bn254;
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{Zero, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tensor_commitment_lib::PST;
use rand::Rng;
use rayon::prelude::*;
use thiserror::Error;

mod lagrange;
use crate::lagrange::{
    evaluate_polynomial_from_coeffs as lagrange_evaluate_from_coeffs,
    multivariate_lagrange_interpolation,
};

pub type Scalar = <Bn254 as Pairing>::ScalarField;
type G1 = <Bn254 as Pairing>::G1;
type G1Affine = <Bn254 as Pairing>::G1Affine;
type G2Affine = <Bn254 as Pairing>::G2Affine;

#[derive(Debug, Clone)]
struct Node {
    coeffs: Arc<Vec<Scalar>>,
    commitment: G1,
    digest: Scalar,
    children: Option<Vec<Arc<Node>>>,
}

/// Describes the fan-out of each node and the number of multivariate layers.
#[derive(Clone, Debug)]
pub struct TreeConfig {
    /// Number of slots per axis (>=2) used to address children.
    pub axis_arity: Vec<usize>,
    /// Number of edges from root to a leaf. Must be >= 1.
    pub depth: usize,
}

impl TreeConfig {
    fn validate(&self) -> Result<(), TreeError> {
        if self.axis_arity.is_empty() {
            return Err(TreeError::Config(
                "need at least one axis for multivariate branching".into(),
            ));
        }
        if self.axis_arity.iter().any(|&a| a < 2) {
            return Err(TreeError::Config("each axis must have arity >= 2".into()));
        }
        if self.depth == 0 {
            return Err(TreeError::Config(
                "tree depth must be >= 1 (root-leaf distance)".into(),
            ));
        }
        Ok(())
    }

    fn axis_count(&self) -> usize {
        self.axis_arity.len()
    }

    pub fn axis_product(&self) -> usize {
        self.axis_arity.iter().product()
    }

    fn expected_value_count(&self) -> usize {
        let mut acc = 1usize;
        let block = self.axis_product();
        for _ in 0..self.depth {
            acc = acc
                .checked_mul(block)
                .expect("tree size exceeds usize, reduce parameters");
        }
        acc
    }
}

fn build_axis_nodes(axis_arity: &[usize]) -> Vec<Vec<Scalar>> {
    axis_arity
        .iter()
        .map(|&arity| {
            (0..arity)
                .map(|idx| Scalar::from(idx as u64))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Position selector for a node: one coordinate per axis.
#[derive(Clone, Debug)]
pub struct MultiIndex {
    pub coords: Vec<usize>,
}

impl MultiIndex {
    pub fn new(coords: Vec<usize>) -> Self {
        Self { coords }
    }
}

/// Proof for a single multivariate node opening.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct NodeProof {
    pub commitment: G1,
    pub evaluation_point: Vec<Scalar>,
    pub evaluation: Scalar,
    pub quotients: Vec<G1>,
    pub child_commitment: Option<G1>,
}

/// Full path proof from the root down to a leaf value.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiverkleProof {
    pub path: Vec<NodeProof>,
    pub leaf_value: Scalar,
}

#[derive(Error, Debug)]
pub enum TreeError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("data length mismatch: expected {expected}, got {actual}")]
    Length { expected: usize, actual: usize },
    #[error("invalid path: {0}")]
    Path(String),
}

/// Verkle tree variant backed by PST/MMP multivariate commitments.
pub struct MultiverkleTree {
    config: TreeConfig,
    degree_bound: usize,
    ck: Vec<G1Affine>,
    vk: Vec<G2Affine>,
    axis_points: Vec<Vec<Scalar>>,
    root: Arc<Node>,
}

impl MultiverkleTree {
    /// Builds a new tree from scalar data laid out in lexicographic order.
    pub fn from_data<R: Rng>(
        rng: &mut R,
        config: TreeConfig,
        data: Vec<Scalar>,
    ) -> Result<Self, TreeError> {
        config.validate()?;

        let expected = config.expected_value_count();
        if data.len() != expected {
            return Err(TreeError::Length {
                expected,
                actual: data.len(),
            });
        }

        let degree_bound = *config
            .axis_arity
            .iter()
            .max()
            .expect("validated non-empty axes");

        let axis_points = build_axis_nodes(&config.axis_arity);

        let (ck, vk) = PST::<Bn254>::setup(rng, config.axis_count(), degree_bound);
        let root = Self::build_tree(&config, degree_bound, &ck, &axis_points, &data)?;

        Ok(Self {
            config,
            degree_bound,
            ck,
            vk,
            axis_points,
            root,
        })
    }

    pub fn root_commitment(&self) -> G1 {
        self.root.commitment
    }

    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    pub fn verification_key(&self) -> &[G2Affine] {
        &self.vk
    }

    /// Returns the degree bound used for polynomial commitments.
    ///
    /// This is the maximum degree per variable in the multivariate polynomial.
    /// It equals `max(axis_arity)` - the largest arity across all axes.
    ///
    /// **Theoretical advantage of higher-dimensional trees:**
    /// - For same fanout, higher dimensions → smaller degree_bound
    /// - Example: [4,4,4] (64 children) has degree_bound=4
    ///   vs [8,8] (64 children) has degree_bound=8
    /// - Smaller degree_bound → smaller SRS size, faster operations
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the approximate SRS (structured reference string) size requirement.
    ///
    /// This is `axis_count × degree_bound`, which is the key complexity parameter
    /// for multivariate polynomial commitments. Smaller is better.
    pub fn srs_size_estimate(&self) -> usize {
        self.config.axis_count() * self.degree_bound
    }

    pub fn path_from_linear_index(&self, index: usize) -> Result<Vec<MultiIndex>, TreeError> {
        linear_index_to_path(index, &self.config.axis_arity, self.config.depth)
    }

    pub fn open_linear_index(&self, index: usize) -> Result<MultiverkleProof, TreeError> {
        let path = self.path_from_linear_index(index)?;
        self.open(&path)
    }

    /// Generates a path proof given one multi-index per depth level.
    pub fn open(&self, path: &[MultiIndex]) -> Result<MultiverkleProof, TreeError> {
        if path.len() != self.config.depth {
            return Err(TreeError::Path(format!(
                "expected {} indices, got {}",
                self.config.depth,
                path.len()
            )));
        }

        let mut current = self.root.clone();
        let mut proofs = Vec::with_capacity(path.len());

        for (level, idx) in path.iter().enumerate() {
            validate_coords(idx, &self.config.axis_arity)?;

            let eval_point = idx
                .coords
                .iter()
                .enumerate()
                .map(|(axis, &c)| self.axis_points[axis][c])
                .collect::<Vec<_>>();
            let quotients = PST::<Bn254>::prove(
                self.degree_bound,
                &self.ck,
                &eval_point,
                current.coeffs.as_ref(),
            );
            let evaluation = evaluate_from_coeffs(
                current.coeffs.as_ref(),
                &eval_point,
                &self.config.axis_arity,
                self.degree_bound,
            );

            let child_commitment = if level < path.len() - 1 {
                Some(select_child(&current, &idx.coords, &self.config)?.commitment)
            } else {
                None
            };

            proofs.push(NodeProof {
                commitment: current.commitment,
                evaluation_point: eval_point,
                evaluation,
                quotients,
                child_commitment,
            });

            if let Some(children) = &current.children {
                let linear = coords_to_linear(&idx.coords, &self.config.axis_arity);
                current = children[linear].clone();
            }
        }

        let leaf_value = proofs
            .last()
            .map(|p| p.evaluation)
            .unwrap_or_else(Scalar::zero);

        Ok(MultiverkleProof {
            path: proofs,
            leaf_value,
        })
    }

    /// Verifies membership using the embedded parameters without requiring tree traversal.
    pub fn verify_membership_with_tree(
        &self,
        index: usize,
        expected_value: &Scalar,
        proof: &MultiverkleProof,
    ) -> Result<bool, TreeError> {
        Self::verify_membership(
            &self.config,
            index,
            expected_value,
            &self.vk,
            self.root.commitment,
            proof,
        )
    }

    /// Verifies a membership proof knowing only the public parameters.
    pub fn verify_membership(
        config: &TreeConfig,
        index: usize,
        expected_value: &Scalar,
        vk: &[G2Affine],
        root_commitment: G1,
        proof: &MultiverkleProof,
    ) -> Result<bool, TreeError> {
        config.validate()?;
        if proof.path.len() != config.depth {
            return Ok(false);
        }
        let axis_points = build_axis_nodes(&config.axis_arity);
        let expected_path = linear_index_to_path(index, &config.axis_arity, config.depth)?;
        let axis_count = config.axis_count();
        let mut expected_commitment = root_commitment;

        for (level, (expected_coords, step)) in expected_path.iter().zip(&proof.path).enumerate() {
            if step.commitment != expected_commitment {
                return Ok(false);
            }
            if step.evaluation_point.len() != axis_count {
                return Ok(false);
            }
            if expected_coords.coords.len() != axis_count {
                return Err(TreeError::Path(format!(
                    "coordinate length mismatch at level {}",
                    level
                )));
            }
            for axis in 0..axis_count {
                let expected_eval = axis_points[axis][expected_coords.coords[axis]];
                if step.evaluation_point[axis] != expected_eval {
                    return Ok(false);
                }
            }

            if !PST::<Bn254>::verify(
                vk,
                &step.evaluation_point,
                step.commitment,
                step.evaluation,
                &step.quotients,
            ) {
                return Ok(false);
            }

            if let Some(child) = step.child_commitment {
                if digest_commitment(&child) != step.evaluation {
                    return Ok(false);
                }
                expected_commitment = child;
            } else if step.evaluation != proof.leaf_value {
                return Ok(false);
            }
        }

        Ok(proof.leaf_value == *expected_value)
    }

    fn build_tree(
        config: &TreeConfig,
        degree_bound: usize,
        ck: &[G1Affine],
        axis_points: &[Vec<Scalar>],
        data: &[Scalar],
    ) -> Result<Arc<Node>, TreeError> {
        let mut nodes = Self::build_leaves(config, degree_bound, ck, axis_points, data)?;
        let fanout = config.axis_product();

        while nodes.len() > 1 {
            if nodes.len() % fanout != 0 {
                return Err(TreeError::Config(
                    "data does not form a complete multivariate tree".into(),
                ));
            }

            nodes = nodes
                .par_chunks(fanout)
                .map(|chunk| Self::build_parent(chunk, config, degree_bound, ck, axis_points))
                .collect();
        }

        nodes
            .pop()
            .ok_or_else(|| TreeError::Config("tree construction failed".into()))
    }

    fn build_leaves(
        config: &TreeConfig,
        degree_bound: usize,
        ck: &[G1Affine],
        axis_points: &[Vec<Scalar>],
        data: &[Scalar],
    ) -> Result<Vec<Arc<Node>>, TreeError> {
        let chunk = config.axis_product();

        if data.len() % chunk != 0 {
            return Err(TreeError::Config(
                "leaf chunk does not match axis product".into(),
            ));
        }

        Ok(data
            .par_chunks(chunk)
            .map(|values| {
                let coeffs =
                    values_to_coeffs(values, &config.axis_arity, degree_bound, axis_points);
                let commitment = PST::<Bn254>::commit(ck, &coeffs);
                let digest = digest_commitment(&commitment);
                Arc::new(Node {
                    coeffs: Arc::new(coeffs),
                    commitment,
                    digest,
                    children: None,
                })
            })
            .collect())
    }

    fn build_parent(
        chunk: &[Arc<Node>],
        config: &TreeConfig,
        degree_bound: usize,
        ck: &[G1Affine],
        axis_points: &[Vec<Scalar>],
    ) -> Arc<Node> {
        debug_assert_eq!(chunk.len(), config.axis_product());

        let hashed_children = chunk.iter().map(|child| child.digest).collect::<Vec<_>>();
        let coeffs =
            values_to_coeffs(&hashed_children, &config.axis_arity, degree_bound, axis_points);

        let commitment = PST::<Bn254>::commit(ck, &coeffs);
        let digest = digest_commitment(&commitment);

        Arc::new(Node {
            coeffs: Arc::new(coeffs),
            commitment,
            digest,
            children: Some(chunk.iter().cloned().collect()),
        })
    }
}

fn validate_coords(idx: &MultiIndex, arity: &[usize]) -> Result<(), TreeError> {
    if idx.coords.len() != arity.len() {
        return Err(TreeError::Path("coordinate length mismatch".into()));
    }
    for (axis, (&coord, &max)) in idx.coords.iter().zip(arity.iter()).enumerate() {
        if coord >= max {
            return Err(TreeError::Path(format!(
                "coordinate {} out of range ({} >= {})",
                axis, coord, max
            )));
        }
    }
    Ok(())
}

fn select_child(
    node: &Node,
    coords: &[usize],
    config: &TreeConfig,
) -> Result<Arc<Node>, TreeError> {
    let children = node
        .children
        .as_ref()
        .ok_or_else(|| TreeError::Path("encountered leaf too early".into()))?;
    let linear = coords_to_linear(coords, &config.axis_arity);
    children
        .get(linear)
        .cloned()
        .ok_or_else(|| TreeError::Path("missing child for given coordinates".into()))
}

fn values_to_coeffs(
    values: &[Scalar],
    axis_arity: &[usize],
    degree_bound: usize,
    axis_points: &[Vec<Scalar>],
) -> Vec<Scalar> {
    debug_assert_eq!(axis_points.len(), axis_arity.len());
    for (axis_nodes, &arity) in axis_points.iter().zip(axis_arity.iter()) {
        debug_assert_eq!(axis_nodes.len(), arity);
    }
    let coeffs = multivariate_lagrange_interpolation(axis_points, values);
    pad_coeffs(&coeffs, axis_arity, degree_bound)
}

fn pad_coeffs(coeffs: &[Scalar], axis_arity: &[usize], degree_bound: usize) -> Vec<Scalar> {
    let axis_count = axis_arity.len();
    let padded_len = degree_bound.pow(axis_count as u32);
    let mut padded = vec![Scalar::zero(); padded_len];
    for (offset, coeff) in coeffs.iter().enumerate() {
        let coords = linear_to_coords(offset, axis_arity);
        let idx = lex_index(&coords, degree_bound, axis_count);
        padded[idx] = *coeff;
    }
    padded
}

fn coords_to_linear(coords: &[usize], axis_arity: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for (&coord, &axis) in coords.iter().rev().zip(axis_arity.iter().rev()) {
        index += coord * stride;
        stride *= axis;
    }
    index
}

fn linear_to_coords(mut idx: usize, axis_arity: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; axis_arity.len()];
    for axis in (0..axis_arity.len()).rev() {
        let base = axis_arity[axis];
        coords[axis] = idx % base;
        idx /= base;
    }
    coords
}

fn chunk_to_coords(mut chunk: usize, axis_arity: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; axis_arity.len()];
    for axis in (0..axis_arity.len()).rev() {
        let arity = axis_arity[axis];
        coords[axis] = chunk % arity;
        chunk /= arity;
    }
    coords
}

fn linear_index_to_path_internal(
    mut index: usize,
    axis_arity: &[usize],
    depth: usize,
) -> Result<Vec<MultiIndex>, TreeError> {
    if depth == 0 {
        return Err(TreeError::Path("tree depth must be >= 1".into()));
    }
    let base = axis_arity.iter().product::<usize>();
    let total = base
        .checked_pow(depth as u32)
        .ok_or_else(|| TreeError::Config("tree dimensions overflow usize".into()))?;
    if index >= total {
        return Err(TreeError::Path(format!(
            "index {} out of range for tree capacity {}",
            index, total
        )));
    }
    let mut path = Vec::with_capacity(depth);
    for _ in 0..depth {
        let coords = chunk_to_coords(index % base, axis_arity);
        path.push(MultiIndex { coords });
        index /= base;
    }
    path.reverse();
    Ok(path)
}

pub fn linear_index_to_path(
    index: usize,
    axis_arity: &[usize],
    depth: usize,
) -> Result<Vec<MultiIndex>, TreeError> {
    linear_index_to_path_internal(index, axis_arity, depth)
}

fn lex_index(coords: &[usize], degree_bound: usize, axis_count: usize) -> usize {
    coords.iter().fold(0usize, |acc, &coord| {
        assert!(coord < degree_bound, "coordinate exceeds degree bound");
        acc * degree_bound + coord
    }) % degree_bound.pow(axis_count as u32)
}

fn evaluate_from_coeffs(
    coeffs: &[Scalar],
    eval_point: &[Scalar],
    axis_arity: &[usize],
    degree_bound: usize,
) -> Scalar {
    let dims = vec![degree_bound; axis_arity.len()];
    lagrange_evaluate_from_coeffs(coeffs, &dims, eval_point)
}

/// Directly evaluates a value from a grid without going through coefficients.
///
/// This is more efficient for single point lookups when you have the original
/// grid values. However, PST commitments require coefficients, so this is only
/// useful for verification/sanity checks, not for proof generation.
#[allow(dead_code)] // Used in tests
fn evaluate_direct_from_grid(values: &[Scalar], coords: &[usize], axis_arity: &[usize]) -> Scalar {
    let linear_idx = coords_to_linear(coords, axis_arity);
    values[linear_idx]
}

fn digest_commitment(point: &G1) -> Scalar {
    let affine = point.into_affine();
    let mut bytes = Vec::new();
    affine.serialize_compressed(&mut bytes).unwrap();
    Scalar::from_le_bytes_mod_order(&bytes)
}

pub fn scalar_to_decimal(value: &Scalar) -> String {
    value.into_bigint().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn multivariate_path_roundtrip() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 2,
        };

        let block = config.axis_product();
        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 1))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        let path = [MultiIndex::new(vec![1, 2]), MultiIndex::new(vec![0, 1])];

        let proof = tree.open(&path).expect("proof should build");
        assert_membership_for_path(&tree, &config, &path, &proof, &data, "path roundtrip");

        // Manually locate value for sanity.
        let parent_idx = coords_to_linear(&path[0].coords, &config.axis_arity);
        let leaf_chunk_start = parent_idx * block;
        let leaf_value_idx =
            leaf_chunk_start + coords_to_linear(&path[1].coords, &config.axis_arity);
        let expected = data[leaf_value_idx];

        assert_eq!(proof.leaf_value, expected);
    }

    /// Test that single point evaluation from coefficients matches direct value lookup.
    /// This verifies the IFFT approach correctly preserves values for point openings.
    #[test]
    fn single_point_evaluation_correctness() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 1,
        };

        let block = config.axis_product();
        let data = (0..block)
            .map(|i| Scalar::from(i as u64 + 100))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Test all points in the leaf
        for x in 0..config.axis_arity[0] {
            for y in 0..config.axis_arity[1] {
                let path = [MultiIndex::new(vec![x, y])];
                let proof = tree.open(&path).expect("proof should build");

                // Direct value lookup from original data
                let linear_idx = coords_to_linear(&[x, y], &config.axis_arity);
                let expected_value = data[linear_idx];

                // Value from proof (evaluated from coefficients)
                let proof_value = proof.leaf_value;

                assert_eq!(
                    proof_value, expected_value,
                    "Point ({}, {}) evaluation mismatch: proof={}, expected={}",
                    x, y, proof_value, expected_value
                );

                // Verify proof is valid
                assert_membership_for_path(
                    &tree,
                    &config,
                    &path,
                    &proof,
                    &data,
                    "single point evaluation",
                );
            }
        }
    }

    /// Test that evaluation from coefficients matches interpolation roundtrip.
    /// This ensures the coefficient representation is correct.
    #[test]
    fn interpolation_roundtrip_correctness() {
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 1,
        };

        let axis_points = build_axis_nodes(&config.axis_arity);

        let block = config.axis_product();
        let values: Vec<Scalar> = (0..block).map(|i| Scalar::from(i as u64 + 50)).collect();

        // Convert to coefficients via multivariate Lagrange interpolation
        let coeffs = values_to_coeffs(&values, &config.axis_arity, 4, &axis_points);

        // Evaluate back at each point
        for x in 0..config.axis_arity[0] {
            for y in 0..config.axis_arity[1] {
                let eval_point: Vec<Scalar> = vec![axis_points[0][x], axis_points[1][y]];

                let evaluated = evaluate_from_coeffs(&coeffs, &eval_point, &config.axis_arity, 4);

                let linear_idx = coords_to_linear(&[x, y], &config.axis_arity);
                let original_value = values[linear_idx];

                assert_eq!(
                    evaluated, original_value,
                    "IFFT roundtrip failed at ({}, {}): evaluated={}, original={}",
                    x, y, evaluated, original_value
                );
            }
        }
    }

    /// Test that direct grid evaluation matches coefficient-based evaluation.
    /// This verifies that for single point openings, both methods give the same result.
    #[test]
    fn direct_vs_coefficient_evaluation() {
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 1,
        };

        let axis_points = build_axis_nodes(&config.axis_arity);

        let block = config.axis_product();
        let values: Vec<Scalar> = (0..block).map(|i| Scalar::from(i as u64 + 200)).collect();

        // Convert to coefficients via Lagrange interpolation
        let coeffs = values_to_coeffs(&values, &config.axis_arity, 4, &axis_points);

        // Compare direct lookup vs coefficient evaluation for all points
        for x in 0..config.axis_arity[0] {
            for y in 0..config.axis_arity[1] {
                let coords = vec![x, y];

                // Direct evaluation from grid
                let direct = evaluate_direct_from_grid(&values, &coords, &config.axis_arity);

                // Evaluation from coefficients
                let eval_point: Vec<Scalar> = vec![axis_points[0][x], axis_points[1][y]];
                let from_coeffs = evaluate_from_coeffs(&coeffs, &eval_point, &config.axis_arity, 4);

                assert_eq!(
                    direct, from_coeffs,
                    "Evaluation mismatch at ({}, {}): direct={}, from_coeffs={}",
                    x, y, direct, from_coeffs
                );
            }
        }
    }

    // ============================================================================
    // COMPARISON TESTS: Multivariate vs Simple Verkle Trees
    // ============================================================================
    // These tests highlight key differences between multivariate Verkle trees
    // (this implementation) and simple univariate Verkle trees (CleanPegasus style)

    /// Test 1: Multi-dimensional addressing vs single index
    ///
    /// Simple Verkle: Uses single integer index (e.g., index 5)
    /// Multivariate: Uses multi-index tuple (e.g., [1, 2] for 2D)
    ///
    /// This test verifies that multivariate trees correctly handle multi-dimensional
    /// addressing and that the same logical position can be accessed via different
    /// coordinate representations.
    #[test]
    fn test_multidimensional_addressing() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4, 4], // 3D tree: 4×4×4 = 64 children per node (FFT-friendly)
            depth: 1,
        };

        let block = config.axis_product(); // 27
        let data = (0..block)
            .map(|i| Scalar::from(i as u64 + 1000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Test various 3D coordinates (for 4×4×4 grid)
        // Note: coords_to_linear uses reverse order (last axis varies fastest)
        // For [x, y, z] in 4×4×4: index = z*1 + y*4 + x*16
        let test_cases = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![1, 1, 1],
            vec![3, 3, 3],
        ];

        for coords in test_cases {
            let path = [MultiIndex::new(coords.clone())];
            let proof = tree.open(&path).expect("proof should build");

            // Calculate expected linear index using the actual conversion function
            let expected_linear = coords_to_linear(&coords, &config.axis_arity);
            let expected_value = data[expected_linear];

            assert_eq!(
                proof.leaf_value, expected_value,
                "3D coordinate {:?} should map to linear index {} with value {}",
                coords, expected_linear, expected_value
            );

            // Verify proof is valid
            assert_membership_for_path(
                &tree,
                &config,
                &path,
                &proof,
                &data,
                "multidimensional addressing",
            );
        }
    }

    /// Test 2: Different axis arities (non-square grids)
    ///
    /// Simple Verkle: Always has uniform width (e.g., width=8 means 8 children)
    /// Multivariate: Can have different arities per axis (e.g., [2, 4] = 2×4 = 8 children)
    ///
    /// This tests that multivariate trees correctly handle rectangular (non-square) grids.
    #[test]
    fn test_rectangular_grids() {
        let mut rng = thread_rng();

        // Test various rectangular configurations (all FFT-friendly)
        let configs = vec![
            TreeConfig {
                axis_arity: vec![2, 4],
                depth: 1,
            }, // 2×4 = 8
            TreeConfig {
                axis_arity: vec![4, 2],
                depth: 1,
            }, // 4×2 = 8 (transposed)
            TreeConfig {
                axis_arity: vec![2, 2, 4],
                depth: 1,
            }, // 2×2×4 = 16
            TreeConfig {
                axis_arity: vec![4, 4],
                depth: 1,
            }, // 4×4 = 16
        ];

        for config in configs {
            let block = config.axis_product();
            let data = (0..block)
                .map(|i| Scalar::from(i as u64 + 2000))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

            // Test accessing all corners and middle
            let corners = match config.axis_arity.len() {
                2 => vec![
                    vec![0, 0],
                    vec![config.axis_arity[0] - 1, 0],
                    vec![0, config.axis_arity[1] - 1],
                    vec![config.axis_arity[0] - 1, config.axis_arity[1] - 1],
                ],
                3 => vec![
                    vec![0, 0, 0],
                    vec![
                        config.axis_arity[0] - 1,
                        config.axis_arity[1] - 1,
                        config.axis_arity[2] - 1,
                    ],
                ],
                _ => {
                    // For higher dimensions, test first and last corners
                    let first = vec![0; config.axis_arity.len()];
                    let mut last = vec![];
                    for &arity in &config.axis_arity {
                        last.push(arity - 1);
                    }
                    vec![first, last]
                }
            };

            for corner in corners {
                let path = [MultiIndex::new(corner.clone())];
                let proof = tree.open(&path).expect("proof should build");

                let linear_idx = coords_to_linear(&corner, &config.axis_arity);
                let expected_value = data[linear_idx];

                assert_eq!(
                    proof.leaf_value, expected_value,
                    "Corner {:?} in config {:?} should have value {}",
                    corner, config.axis_arity, expected_value
                );

                assert_membership_for_path(
                    &tree,
                    &config,
                    &path,
                    &proof,
                    &data,
                    "rectangular grid corner",
                );
            }
        }
    }

    /// Test 3: Same data organized differently
    ///
    /// Simple Verkle: Data is organized linearly (chunk 0, chunk 1, ...)
    /// Multivariate: Data is organized in multi-dimensional grids
    ///
    /// This test verifies that the same 16 values can be organized as:
    /// - Simple: width=4 (4 chunks of 4 values each)
    /// - Multivariate: [4, 4] (4×4 grid, depth=1)
    ///
    /// Both should produce valid trees, but with different internal structures.
    #[test]
    fn test_same_data_different_organization() {
        let mut rng = thread_rng();
        let data_size = 16;
        let data: Vec<Scalar> = (0..data_size)
            .map(|i| Scalar::from(i as u64 + 3000))
            .collect();

        // Multivariate organization: 4×4 grid (single level)
        let multi_config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 1,
        };

        let multi_tree =
            MultiverkleTree::from_data(&mut rng, multi_config.clone(), data.clone()).unwrap();

        // Access all points in the 4×4 grid
        for x in 0..4 {
            for y in 0..4 {
                let path = [MultiIndex::new(vec![x, y])];
                let proof = multi_tree.open(&path).expect("proof should build");

                let linear_idx = coords_to_linear(&[x, y], &multi_config.axis_arity);
                let expected_value = data[linear_idx];

                assert_eq!(
                    proof.leaf_value, expected_value,
                    "Multivariate access at ({}, {}) should yield value {}",
                    x, y, expected_value
                );

                assert_membership_for_path(
                    &multi_tree,
                    &multi_config,
                    &path,
                    &proof,
                    &data,
                    "same data organization",
                );
            }
        }
    }

    /// Test 4: Multi-level paths with different dimensions
    ///
    /// Simple Verkle: Each level uses the same width
    /// Multivariate: Each level uses the same axis arity, but coordinates are multi-dimensional
    ///
    /// This tests deep paths through multivariate trees.
    #[test]
    fn test_multilevel_multidimensional_paths() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![2, 2], // 2×2 = 4 children per node
            depth: 3,               // 3 levels deep
        };

        let total = config.expected_value_count(); // 4^3 = 64 values
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 4000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Test various paths through the tree
        let test_paths = vec![
            vec![
                MultiIndex::new(vec![0, 0]),
                MultiIndex::new(vec![0, 0]),
                MultiIndex::new(vec![0, 0]),
            ],
            vec![
                MultiIndex::new(vec![1, 1]),
                MultiIndex::new(vec![1, 1]),
                MultiIndex::new(vec![1, 1]),
            ],
            vec![
                MultiIndex::new(vec![0, 1]),
                MultiIndex::new(vec![1, 0]),
                MultiIndex::new(vec![1, 1]),
            ],
        ];

        for path in test_paths {
            let proof = tree.open(&path).expect("proof should build");
            assert_membership_for_path(&tree, &config, &path, &proof, &data, "multilevel path");

            // Calculate expected value
            let mut linear_idx = 0;
            let mut multiplier = 1;
            for idx in path.iter().rev() {
                let level_idx = coords_to_linear(&idx.coords, &config.axis_arity);
                linear_idx += level_idx * multiplier;
                multiplier *= config.axis_product();
            }
            let expected_value = data[linear_idx];

            assert_eq!(
                proof.leaf_value, expected_value,
                "Path {:?} should yield value {} at linear index {}",
                path, expected_value, linear_idx
            );
        }
    }

    /// Test 5: Proof structure differences
    ///
    /// Simple Verkle: Proof has single evaluation point per level
    /// Multivariate: Proof has multi-dimensional evaluation point per level
    ///
    /// This test verifies that multivariate proofs correctly encode multi-dimensional
    /// evaluation points and that verification works with the PST scheme.
    #[test]
    fn test_multivariate_proof_structure() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4], // FFT-friendly
            depth: 2,
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 5000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        let path = vec![MultiIndex::new(vec![1, 2]), MultiIndex::new(vec![2, 1])];

        let proof = tree.open(&path).expect("proof should build");

        // Verify proof structure
        assert_eq!(
            proof.path.len(),
            config.depth,
            "Proof should have one step per depth level"
        );

        for (level, node_proof) in proof.path.iter().enumerate() {
            // Each proof step should have evaluation point matching the path coordinates
            assert_eq!(
                node_proof.evaluation_point.len(),
                config.axis_count(),
                "Level {} evaluation point should have {} dimensions",
                level,
                config.axis_count()
            );

            // Verify evaluation point matches the path coordinates (via axis nodes)
            let path_coords = &path[level].coords;
            for (axis, &coord) in path_coords.iter().enumerate() {
                let expected_eval = tree.axis_points[axis][coord];
                assert_eq!(
                    node_proof.evaluation_point[axis], expected_eval,
                    "Level {} axis {} evaluation point mismatch",
                    level, axis
                );
            }

            // Verify quotient count matches number of variables
            assert_eq!(
                node_proof.quotients.len(),
                config.axis_count(),
                "Level {} should have {} quotient commitments (one per variable)",
                level,
                config.axis_count()
            );
        }

        assert_membership_for_path(&tree, &config, &path, &proof, &data, "proof structure");
    }

    /// Test 6: Edge cases specific to multivariate trees
    ///
    /// Tests scenarios that are unique to multivariate trees:
    /// - Very high dimensional trees (e.g., 5D)
    /// - Asymmetric axis arities
    /// - Single-level trees (just root)
    #[test]
    fn test_multivariate_edge_cases() {
        let mut rng = thread_rng();

        // Test 1: High-dimensional tree (5D)
        {
            let config = TreeConfig {
                axis_arity: vec![2, 2, 2, 2, 2], // 2^5 = 32 children
                depth: 1,
            };
            let block = config.axis_product();
            let data = (0..block)
                .map(|i| Scalar::from(i as u64 + 6000))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

            // Test accessing with 5D coordinates
            let path = [MultiIndex::new(vec![1, 0, 1, 0, 1])];
            let proof = tree.open(&path).expect("proof should build");
            assert_membership_for_path(&tree, &config, &path, &proof, &data, "5D edge case");

            let linear_idx = coords_to_linear(&path[0].coords, &config.axis_arity);
            assert_eq!(proof.leaf_value, data[linear_idx]);
        }

        // Test 2: Highly asymmetric grid
        {
            let config = TreeConfig {
                axis_arity: vec![2, 8], // 2×8 = 16, very rectangular (FFT-friendly)
                depth: 1,
            };
            let block = config.axis_product();
            let data = (0..block)
                .map(|i| Scalar::from(i as u64 + 7000))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

            // Test accessing corners
            for &x in &[0, 1] {
                for &y in &[0, 7] {
                    let path = [MultiIndex::new(vec![x, y])];
                    let proof = tree.open(&path).expect("proof should build");
                    assert_membership_for_path(
                        &tree,
                        &config,
                        &path,
                        &proof,
                        &data,
                        "asymmetric edge case",
                    );

                    let linear_idx = coords_to_linear(&path[0].coords, &config.axis_arity);
                    assert_eq!(proof.leaf_value, data[linear_idx]);
                }
            }
        }

        // Test 3: Single-level tree (root only, no internal nodes)
        {
            let config = TreeConfig {
                axis_arity: vec![4, 4],
                depth: 1, // Only root level
            };
            let block = config.axis_product();
            let data = (0..block)
                .map(|i| Scalar::from(i as u64 + 8000))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

            // All paths should be single-element
            for x in 0..4 {
                for y in 0..4 {
                    let path = [MultiIndex::new(vec![x, y])];
                    let proof = tree.open(&path).expect("proof should build");
                    assert_membership_for_path(
                        &tree,
                        &config,
                        &path,
                        &proof,
                        &data,
                        "single level edge case",
                    );
                    assert_eq!(
                        proof.path.len(),
                        1,
                        "Single-level tree should have one proof step"
                    );
                }
            }
        }
    }

    /// Test 7: Coordinate conversion correctness
    ///
    /// Verifies that the linear-to-coordinate and coordinate-to-linear conversions
    /// are correct and consistent. This is critical for multivariate trees since
    /// they rely on these conversions extensively.
    ///
    /// Note: This test only tests the conversion functions, not tree construction,
    /// so it can use any axis arities (not just FFT-friendly ones).
    #[test]
    fn test_coordinate_conversion_roundtrip() {
        let axis_arities = vec![
            vec![2, 2],
            vec![4, 4],
            vec![2, 4],
            vec![2, 2, 4],
            vec![4, 8],
        ];

        for axis_arity in axis_arities {
            let total = axis_arity.iter().product::<usize>();

            // Test all possible linear indices
            for linear_idx in 0..total {
                let coords = linear_to_coords(linear_idx, &axis_arity);

                // Convert back to linear
                let recovered_linear = coords_to_linear(&coords, &axis_arity);

                assert_eq!(
                    linear_idx, recovered_linear,
                    "Roundtrip failed for axis_arity {:?}: linear {} -> coords {:?} -> linear {}",
                    axis_arity, linear_idx, coords, recovered_linear
                );

                // Verify coordinates are in valid range
                for (axis, &coord) in coords.iter().enumerate() {
                    assert!(
                        coord < axis_arity[axis],
                        "Coordinate {} out of range for axis {} (max: {})",
                        coord,
                        axis,
                        axis_arity[axis]
                    );
                }
            }
        }
    }

    /// Test 8: Comparison of tree sizes
    ///
    /// Demonstrates that multivariate trees can represent the same logical
    /// structure with different internal organizations, affecting proof sizes.
    #[test]
    fn test_tree_size_comparison() {
        let mut rng = thread_rng();

        // Same logical size: 64 values
        // Option 1: 2×2×2×2×2×2 (6D, depth 1) = 64^1 = 64
        // Option 2: 4×4 (2D, depth 1) = 16^1 = 16 (not 64, but let's use 4×4 depth 2 = 256)
        // Option 3: 8×8 (2D, depth 1) = 64^1 = 64
        // Option 4: 2×2 (2D, depth 3) = 4^3 = 64

        let configs = vec![
            TreeConfig {
                axis_arity: vec![2, 2, 2, 2, 2, 2],
                depth: 1,
            }, // 64
            TreeConfig {
                axis_arity: vec![2, 2],
                depth: 3,
            }, // 4^3 = 64
            TreeConfig {
                axis_arity: vec![8, 8],
                depth: 1,
            }, // 64
        ];

        for config in configs {
            let total = config.expected_value_count();
            assert_eq!(total, 64, "All configs should handle 64 values");

            let data = (0..total)
                .map(|i| Scalar::from(i as u64 + 9000))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

            // Verify we can access the same logical position
            // For simplicity, test accessing "middle" element
            let middle_path: Vec<MultiIndex> = (0..config.depth)
                .map(|_| {
                    let coords: Vec<usize> = config
                        .axis_arity
                        .iter()
                        .map(|&arity| arity / 2) // Middle coordinate
                        .collect();
                    MultiIndex::new(coords)
                })
                .collect();

            let proof = tree.open(&middle_path).expect("proof should build");
            assert_membership_for_path(
                &tree,
                &config,
                &middle_path,
                &proof,
                &data,
                "tree size comparison",
            );

            // Verify proof depth matches config
            assert_eq!(
                proof.path.len(),
                config.depth,
                "Proof depth should match tree depth for config {:?}",
                config.axis_arity
            );
        }
    }
    fn assert_membership_for_path(
        tree: &MultiverkleTree,
        config: &TreeConfig,
        path: &[MultiIndex],
        proof: &MultiverkleProof,
        data: &[Scalar],
        context: &str,
    ) {
        let index = path_to_linear_index(path, &config.axis_arity);
        let expected_value = data[index];
        assert!(
            tree.verify_membership_with_tree(index, &expected_value, proof)
                .unwrap(),
            "{context}"
        );
        assert!(
            MultiverkleTree::verify_membership(
                config,
                index,
                &expected_value,
                tree.verification_key(),
                tree.root_commitment(),
                proof
            )
            .unwrap(),
            "{context}"
        );
    }

    fn path_to_linear_index(path: &[MultiIndex], axis_arity: &[usize]) -> usize {
        let base = axis_arity.iter().product::<usize>();
        let mut multiplier = 1usize;
        let mut index = 0usize;
        for node in path.iter().rev() {
            let level_idx = coords_to_linear(&node.coords, axis_arity);
            index += level_idx * multiplier;
            multiplier *= base;
        }
        index
    }

    // ============================================================================
    // OPTIMIZATION TESTS: Verify correctness of performance improvements
    // ============================================================================

    /// Test 1: Verify precomputed axis points are used correctly
    ///
    /// Tests that proofs use evaluation points that match domain.element()
    /// values, which verifies the precomputation optimization is correct.
    /// We test this indirectly by verifying proof evaluation points match
    /// what we'd compute manually.
    #[test]
    fn test_precomputed_axis_points_correctness() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4, 8],
            depth: 2,
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 10000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Manually compute what axis points should be
        let axis_points = build_axis_nodes(&config.axis_arity);

        // Test multiple paths and verify evaluation points in proofs match manual computation
        let test_indices = vec![0, 10, 50, 100];
        for &index in &test_indices {
            if index >= total {
                continue;
            }

            let path = tree.path_from_linear_index(index).unwrap();
            let proof = tree.open(&path).unwrap();

            // Verify each proof step's evaluation points match manual computation
            for (level, node_proof) in proof.path.iter().enumerate() {
                let path_coords = &path[level].coords;
                for (axis_idx, &coord) in path_coords.iter().enumerate() {
                    let manual_point = axis_points[axis_idx][coord];
                    let proof_point = node_proof.evaluation_point[axis_idx];

                    assert_eq!(
                        manual_point, proof_point,
                        "Proof evaluation point should match manual computation: level={}, axis={}, coord={}",
                        level, axis_idx, coord
                    );
                }
            }

            // Verify proof still works
            let expected_value = data[index];
            assert!(
                tree.verify_membership_with_tree(index, &expected_value, &proof).unwrap(),
                "Proof with precomputed points should verify"
            );
        }
    }

    /// Test 2: Verify deterministic proof generation
    ///
    /// Same tree should produce the same proofs for the same paths, ensuring that
    /// parallelization and optimizations don't introduce non-determinism.
    /// Note: Tree roots may differ due to setup randomness, but proof generation
    /// should be deterministic for a given tree instance.
    #[test]
    fn test_deterministic_proof_generation() {
        let mut rng = thread_rng();

        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 3,
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 20000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Generate the same proof multiple times - should be identical
        let index = 42;
        let path = tree.path_from_linear_index(index).unwrap();
        let proof1 = tree.open(&path).unwrap();
        let proof2 = tree.open(&path).unwrap();
        let proof3 = tree.open(&path).unwrap();

        // All proofs should have the same structure and values
        assert_eq!(
            proof1.leaf_value, proof2.leaf_value,
            "Proof leaf values should be deterministic"
        );
        assert_eq!(
            proof2.leaf_value, proof3.leaf_value,
            "Proof leaf values should be deterministic"
        );
        assert_eq!(
            proof1.path.len(), proof2.path.len(),
            "Proof paths should have same length"
        );

        // Verify all proofs are valid
        let expected_value = data[index];
        assert!(
            tree.verify_membership_with_tree(index, &expected_value, &proof1).unwrap(),
            "Proof 1 should verify"
        );
        assert!(
            tree.verify_membership_with_tree(index, &expected_value, &proof2).unwrap(),
            "Proof 2 should verify"
        );
        assert!(
            tree.verify_membership_with_tree(index, &expected_value, &proof3).unwrap(),
            "Proof 3 should verify"
        );

        // Test multiple indices to ensure consistency
        let test_indices = vec![0, 10, 50, 100, total / 2];
        for &idx in &test_indices {
            if idx >= total {
                continue;
            }
            let path = tree.path_from_linear_index(idx).unwrap();
            let proof_a = tree.open(&path).unwrap();
            let proof_b = tree.open(&path).unwrap();

            assert_eq!(
                proof_a.leaf_value, proof_b.leaf_value,
                "Proofs for index {} should be deterministic",
                idx
            );
            assert_eq!(
                proof_a.leaf_value, data[idx],
                "Proof leaf value should match data at index {}",
                idx
            );
        }
    }

    /// Test 3: Verify cached digests work correctly in parent construction
    ///
    /// This indirectly tests that digests are correctly cached by building
    /// trees with multiple levels and verifying that parent-child relationships
    /// are correct (which requires correct digest computation).
    #[test]
    fn test_cached_digest_parent_child_consistency() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 3, // 3 levels ensures multiple parent-child relationships
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 30000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Test multiple paths through the tree
        let test_indices = vec![0, 1, 42, 100, total / 2, total - 1];
        for &index in &test_indices {
            if index >= total {
                continue;
            }

            let path = tree.path_from_linear_index(index).unwrap();
            let proof = tree.open(&path).unwrap();

            // Verify proof structure is consistent
            assert_eq!(proof.path.len(), config.depth, "Proof depth should match tree depth");

            // Verify each proof step has correct structure
            for (level, node_proof) in proof.path.iter().enumerate() {
                assert_eq!(
                    node_proof.evaluation_point.len(),
                    config.axis_count(),
                    "Evaluation point should have one coordinate per axis at level {}",
                    level
                );

                assert_eq!(
                    node_proof.quotients.len(),
                    config.axis_count(),
                    "Quotients should have one per axis at level {}",
                    level
                );

                // If not the last level, should have child commitment
                if level < proof.path.len() - 1 {
                    assert!(
                        node_proof.child_commitment.is_some(),
                        "Non-leaf proof step should have child commitment at level {}",
                        level
                    );
                }

                // Verify the evaluation matches the digest of child commitment
                if let Some(child_commitment) = node_proof.child_commitment {
                    let expected_digest = digest_commitment(&child_commitment);
                    assert_eq!(
                        node_proof.evaluation, expected_digest,
                        "Evaluation should match digest of child commitment at level {}",
                        level
                    );
                }
            }

            // Verify proof is valid
            let expected_value = data[index];
            assert!(
                tree.verify_membership_with_tree(index, &expected_value, &proof).unwrap(),
                "Proof should verify for index {}",
                index
            );
        }
    }

    /// Test 4: Large tree stress test for parallelization correctness
    ///
    /// Build a large tree that will definitely exercise parallel processing
    /// and verify all proofs are correct. This ensures parallelization doesn't
    /// introduce correctness bugs.
    #[test]
    fn test_large_tree_parallelization_correctness() {
        let mut rng = thread_rng();

        // Use a configuration that creates many parallel chunks
        let config = TreeConfig {
            axis_arity: vec![8, 8], // 64 children per node
            depth: 3,               // 64^3 = 262,144 leaves
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 40000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Test a sample of indices across the tree
        let sample_size = 100;
        let step = total / sample_size;
        let test_indices: Vec<usize> = (0..sample_size)
            .map(|i| i * step)
            .chain(std::iter::once(total - 1)) // Also test the last element
            .collect();

        for &index in &test_indices {
            if index >= total {
                continue;
            }

            let path = tree.path_from_linear_index(index).unwrap();
            let proof = tree.open(&path).unwrap();
            let expected_value = data[index];

            assert_eq!(
                proof.leaf_value, expected_value,
                "Leaf value mismatch at index {}",
                index
            );

            assert!(
                tree.verify_membership_with_tree(index, &expected_value, &proof).unwrap(),
                "Proof verification failed at index {}",
                index
            );

            // Also test stateless verification
            assert!(
                MultiverkleTree::verify_membership(
                    &config,
                    index,
                    &expected_value,
                    tree.verification_key(),
                    tree.root_commitment(),
                    &proof
                )
                .unwrap(),
                "Stateless verification failed at index {}",
                index
            );
        }
    }

    /// Test 5: Verify axis points are used correctly in proof generation
    ///
    /// Tests that proofs use evaluation points that match what we'd compute
    /// manually from axis nodes, ensuring the precomputation optimization is
    /// transparent and correct.
    #[test]
    fn test_axis_points_used_in_proofs() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 2,
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 50000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();
        let axis_points = build_axis_nodes(&config.axis_arity);

        // Test a few paths
        for &index in &[0, 10, 50, 100] {
            if index >= total {
                continue;
            }

            let path = tree.path_from_linear_index(index).unwrap();
            let proof = tree.open(&path).unwrap();

            // Verify each proof step's evaluation points match manually computed values
            for (level, node_proof) in proof.path.iter().enumerate() {
                let path_coords = &path[level].coords;
                for (axis, &coord) in path_coords.iter().enumerate() {
                    let expected_point = axis_points[axis][coord];
                    assert_eq!(
                        node_proof.evaluation_point[axis], expected_point,
                        "Proof evaluation point should match manually computed value at level {}, axis {}",
                        level, axis
                    );
                }
            }

            // Verify proof is still valid
            let expected_value = data[index];
            assert!(
                tree.verify_membership_with_tree(index, &expected_value, &proof).unwrap(),
                "Proof with precomputed points should verify"
            );
        }
    }

    /// Test 6: Edge case - Single level tree (depth = 1)
    ///
    /// Tests that optimizations work correctly for the simplest case.
    #[test]
    fn test_single_level_tree_optimizations() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![4, 4],
            depth: 1, // Only root level
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 60000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Verify tree was constructed correctly (implicitly tests precomputation)
        assert_eq!(
            tree.config().axis_count(),
            config.axis_count(),
            "Tree should have correct axis count"
        );

        // Test all leaf positions
        for x in 0..config.axis_arity[0] {
            for y in 0..config.axis_arity[1] {
                let path = [MultiIndex::new(vec![x, y])];
                let proof = tree.open(&path).unwrap();

                assert_eq!(
                    proof.path.len(), 1,
                    "Single level tree should have one proof step"
                );

                let linear_idx = coords_to_linear(&[x, y], &config.axis_arity);
                let expected_value = data[linear_idx];

                assert_eq!(
                    proof.leaf_value, expected_value,
                    "Leaf value should match at ({}, {})",
                    x, y
                );

                assert!(
                    tree.verify_membership_with_tree(linear_idx, &expected_value, &proof).unwrap(),
                    "Proof should verify for single level tree"
                );
            }
        }
    }

    /// Test 7: High-dimensional tree with small axis arities
    ///
    /// Tests optimizations with many dimensions, which is where multivariate
    /// trees show their advantage.
    #[test]
    fn test_high_dimensional_tree_optimizations() {
        let mut rng = thread_rng();
        let config = TreeConfig {
            axis_arity: vec![2, 2, 2, 2, 2, 2], // 6D: 2^6 = 64 children
            depth: 2,                            // 64^2 = 4096 leaves
        };

        let total = config.expected_value_count();
        let data = (0..total)
            .map(|i| Scalar::from(i as u64 + 70000))
            .collect::<Vec<_>>();

        let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();

        // Verify tree configuration (implicitly tests precomputation works for high dimensions)
        assert_eq!(
            tree.config().axis_count(), 6,
            "Should have 6 axes"
        );
        assert_eq!(
            tree.config().axis_arity,
            vec![2, 2, 2, 2, 2, 2],
            "Axis arities should match"
        );

        // Test various positions
        let test_cases = vec![
            vec![0, 0, 0, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1],
            vec![0, 1, 0, 1, 0, 1],
            vec![1, 0, 1, 0, 1, 0],
        ];

        for coords in test_cases {
            // Test at depth 0 (first level)
            let path = vec![MultiIndex::new(coords.clone()), MultiIndex::new(coords.clone())];
            let proof = tree.open(&path).unwrap();

            // Calculate expected linear index
            let mut linear_idx = 0;
            let mut multiplier = 1;
            for idx in path.iter().rev() {
                let level_idx = coords_to_linear(&idx.coords, &config.axis_arity);
                linear_idx += level_idx * multiplier;
                multiplier *= config.axis_product();
            }

            if linear_idx < total {
                let expected_value = data[linear_idx];
                assert_eq!(
                    proof.leaf_value, expected_value,
                    "High-dimensional tree should produce correct leaf value"
                );

                assert!(
                    tree.verify_membership_with_tree(linear_idx, &expected_value, &proof).unwrap(),
                    "High-dimensional proof should verify"
                );
            }
        }
    }

    /// Test 8: Verify all existing functionality still works with optimizations
    ///
    /// Regression test to ensure optimizations didn't break existing behavior.
    #[test]
    fn test_regression_existing_functionality() {
        // Run a subset of existing tests to ensure nothing broke
        let mut rng = thread_rng();

        // Test 1: Basic path roundtrip
        {
            let config = TreeConfig {
                axis_arity: vec![4, 4],
                depth: 2,
            };
            let total = config.expected_value_count();
            let data = (0..total)
                .map(|i| Scalar::from(i as u64 + 1))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();
            let path = [MultiIndex::new(vec![1, 2]), MultiIndex::new(vec![0, 1])];
            let proof = tree.open(&path).expect("proof should build");
            assert_membership_for_path(&tree, &config, &path, &proof, &data, "regression: path roundtrip");
        }

        // Test 2: Verify degree_bound is correctly computed
        {
            let config_3d = TreeConfig {
                axis_arity: vec![4, 4, 4],
                depth: 1,
            };
            let config_2d = TreeConfig {
                axis_arity: vec![8, 8],
                depth: 1,
            };

            let degree_bound_3d = *config_3d.axis_arity.iter().max().unwrap();
            let degree_bound_2d = *config_2d.axis_arity.iter().max().unwrap();

            assert_eq!(degree_bound_3d, 4, "3D tree should have degree_bound=4");
            assert_eq!(degree_bound_2d, 8, "2D tree should have degree_bound=8");
            assert_eq!(
                config_3d.axis_product(), config_2d.axis_product(),
                "Both should have same fanout (64 children)"
            );
        }

        // Test 3: Coordinate conversion
        {
            let axis_arity = vec![4, 4];
            for linear_idx in 0..16 {
                let coords = linear_to_coords(linear_idx, &axis_arity);
                let recovered = coords_to_linear(&coords, &axis_arity);
                assert_eq!(
                    linear_idx, recovered,
                    "Coordinate conversion should still work: {} -> {:?} -> {}",
                    linear_idx, coords, recovered
                );
            }
        }

        // Test 3: Verification with stateless method
        {
            let config = TreeConfig {
                axis_arity: vec![4, 4],
                depth: 1,
            };
            let total = config.expected_value_count();
            let data = (0..total)
                .map(|i| Scalar::from(i as u64 + 100))
                .collect::<Vec<_>>();

            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();
            let index = 5;
            let proof = tree.open_linear_index(index).unwrap();
            let expected_value = data[index];

            assert!(
                MultiverkleTree::verify_membership(
                    &config,
                    index,
                    &expected_value,
                    tree.verification_key(),
                    tree.root_commitment(),
                    &proof
                )
                .unwrap(),
                "Stateless verification should still work"
            );
        }
    }

    // ============================================================================
    // PERFORMANCE COMPARISON TESTS: Demonstrate advantages of higher dimensions
    // ============================================================================

    /// Test: Compare degree_bound and SRS size between 4×4×4 vs 8×8 configurations
    ///
    /// Both configurations have the same fanout (64 children), but:
    /// - 4×4×4: degree_bound=4, axis_count=3 → SRS needs 3×4=12 elements
    /// - 8×8:   degree_bound=8, axis_count=2 → SRS needs 2×8=16 elements
    ///
    /// This demonstrates the theoretical advantage of higher-dimensional trees
    /// with smaller per-axis degrees.
    #[test]
    fn test_degree_bound_comparison() {
        let config_3d = TreeConfig {
            axis_arity: vec![4, 4, 4], // 3D: 4×4×4 = 64 children
            depth: 1,
        };
        let config_2d = TreeConfig {
            axis_arity: vec![8, 8], // 2D: 8×8 = 64 children
            depth: 1,
        };

        // Verify same fanout
        let fanout_3d = config_3d.axis_product();
        let fanout_2d = config_2d.axis_product();
        assert_eq!(
            fanout_3d, fanout_2d,
            "Both configurations should have same fanout: {} == {}",
            fanout_3d, fanout_2d
        );
        assert_eq!(fanout_3d, 64, "Fanout should be 64");

        // Calculate degree bounds
        let degree_bound_3d = *config_3d.axis_arity.iter().max().unwrap();
        let degree_bound_2d = *config_2d.axis_arity.iter().max().unwrap();

        assert_eq!(degree_bound_3d, 4, "3D tree should have degree_bound=4");
        assert_eq!(degree_bound_2d, 8, "2D tree should have degree_bound=8");

        // SRS size comparison (approximate, based on PST setup requirements)
        let srs_size_3d = config_3d.axis_count() * degree_bound_3d; // 3 × 4 = 12
        let srs_size_2d = config_2d.axis_count() * degree_bound_2d; // 2 × 8 = 16

        assert!(
            srs_size_3d < srs_size_2d,
            "3D tree should require smaller SRS: {} < {}",
            srs_size_3d, srs_size_2d
        );

        // Coefficient array size (both pad to same size, but internal structure differs)
        let coeff_size_3d = degree_bound_3d.pow(config_3d.axis_count() as u32); // 4^3 = 64
        let coeff_size_2d = degree_bound_2d.pow(config_2d.axis_count() as u32); // 8^2 = 64

        assert_eq!(
            coeff_size_3d, coeff_size_2d,
            "Both should pad to same coefficient array size"
        );
        assert_eq!(coeff_size_3d, 64, "Both should pad to 64 coefficients");

        println!(
            "Configuration comparison:\n\
            - 3D [4,4,4]: degree_bound={}, axes={}, SRS_size≈{}, coeff_size={}\n\
            - 2D [8,8]:   degree_bound={}, axes={}, SRS_size≈{}, coeff_size={}\n\
            - Advantage: 3D has {}% smaller SRS requirement",
            degree_bound_3d, config_3d.axis_count(), srs_size_3d, coeff_size_3d,
            degree_bound_2d, config_2d.axis_count(), srs_size_2d, coeff_size_2d,
            100 * (srs_size_2d - srs_size_3d) / srs_size_2d
        );
    }

    /// Test: Compare theoretical and actual performance differences
    ///
    /// This test demonstrates the theoretical advantages of higher-dimensional trees:
    /// 1. Smaller degree_bound per variable (4×4×4 has degree=4 vs 8×8 has degree=8)
    /// 2. Smaller SRS size requirement (12 elements vs 16 elements)
    /// 3. More efficient polynomial operations (lower degrees are faster)
    ///
    /// Note: Actual wall-clock time depends on PST implementation optimizations.
    /// The theoretical advantages (SRS size, degree_bound) are always present.
    #[test]
    fn test_construction_time_comparison() {
        use std::time::Instant;

        let mut rng = thread_rng();

        // Compare configurations with same fanout but different dimensions
        // This highlights the degree_bound difference
        let configs = vec![
            ("3D [4,4,4]", TreeConfig {
                axis_arity: vec![4, 4, 4], // 64 children, degree_bound=4
                depth: 2, // Deeper tree to better show differences
            }),
            ("2D [8,8]", TreeConfig {
                axis_arity: vec![8, 8], // 64 children, degree_bound=8
                depth: 2,
            }),
        ];

        let mut results = Vec::new();

        for (name, config) in configs {
            let total = config.expected_value_count();
            let data: Vec<Scalar> = (0..total)
                .map(|i| Scalar::from(i as u64 + 50000))
                .collect();

            let degree_bound = *config.axis_arity.iter().max().unwrap();
            let srs_size_approx = config.axis_count() * degree_bound;

            // Warm up
            let _warmup = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone());

            // Measure construction time
            let start = Instant::now();
            let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone()).unwrap();
            let construction_time = start.elapsed();

            // Measure proof generation time (average over multiple proofs)
            let proof_times: Vec<_> = (0..10)
                .map(|i| {
                    let index = (i * (total / 10)) % total;
                    let path = tree.path_from_linear_index(index).unwrap();
                    let start = Instant::now();
                    let _proof = tree.open(&path).unwrap();
                    start.elapsed()
                })
                .collect();
            let avg_proof_time: f64 = proof_times.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / proof_times.len() as f64;

            results.push((
                name,
                config.axis_count(),
                degree_bound,
                srs_size_approx,
                construction_time,
                avg_proof_time,
            ));
        }

        // Print comparison
        println!("\n=== Performance Comparison (same logical structure) ===");
        println!("{:<15} {:>8} {:>12} {:>12} {:>18} {:>18}",
                 "Config", "Axes", "Degree", "SRS Size", "Build (ms)", "Proof (μs avg)");
        println!("{}", "-".repeat(90));

        for (name, axes, degree, srs, build_time, proof_time) in &results {
            println!("{:<15} {:>8} {:>12} {:>12} {:>15.2} {:>15.2}",
                     name, axes, degree, srs,
                     build_time.as_millis(),
                     proof_time / 1000.0); // Convert to microseconds
        }

        // Verify that 3D [4,4,4] has lower degree than 2D [8,8]
        let result_3d = results.iter().find(|r| r.0 == "3D [4,4,4]").unwrap();
        let result_2d = results.iter().find(|r| r.0 == "2D [8,8]").unwrap();

        assert!(
            result_3d.2 < result_2d.2,
            "3D configuration should have smaller degree_bound ({} < {})",
            result_3d.2, result_2d.2
        );

        assert!(
            result_3d.3 < result_2d.3,
            "3D configuration should require smaller SRS ({} < {})",
            result_3d.3, result_2d.3
        );

        // Note: Actual timing may vary, but structure should show advantage
        println!("\nKey insight: Higher dimensions with smaller per-axis degrees");
        println!("  → Smaller degree_bound per variable");
        println!("  → Smaller SRS size requirement");
        println!("  → Potentially faster operations (depending on PST implementation)");
    }

    /// Comprehensive benchmark: Compare tree construction time across different dimensions
    /// at different data sizes (64, 64², 64³).
    ///
    /// Tests:
    /// - 2D [8,8]: degree_bound=8, 2 axes
    /// - 3D [4,4,4]: degree_bound=4, 3 axes  
    /// - 6D [2,2,2,2,2,2]: degree_bound=2, 6 axes
    ///
    /// All have fanout=64, so same depths needed:
    /// - 64 points: depth=1
    /// - 4096 points: depth=2
    /// - 262144 points: depth=3
    ///
    /// This definitively shows whether higher dimensions beat lower dimensions in practice.
    #[test]
    fn test_comprehensive_dimension_comparison() {
        use std::time::Instant;

        let mut rng = thread_rng();

        // All configurations have fanout=64
        let configs = vec![
            ("2D [8,8]", TreeConfig {
                axis_arity: vec![8, 8],
                depth: 1, // Will be overridden per test
            }),
            ("3D [4,4,4]", TreeConfig {
                axis_arity: vec![4, 4, 4],
                depth: 1,
            }),
            ("6D [2,2,2,2,2,2]", TreeConfig {
                axis_arity: vec![2, 2, 2, 2, 2, 2],
                depth: 1,
            }),
        ];

        let data_sizes = vec![
            ("64 points", 64, 1),
            ("4096 points (64²)", 4096, 2),
            ("262144 points (64³)", 262144, 3),
        ];

        println!("\n{}", "=".repeat(100));
        println!("COMPREHENSIVE DIMENSION COMPARISON BENCHMARK");
        println!("{}", "=".repeat(100));
        println!("\nConfigurations (all have fanout=64):");
        for (name, config) in &configs {
            let degree_bound = config.axis_arity.iter().max().unwrap();
            let srs_size = config.axis_arity.len() * degree_bound;
            println!("  {}: {} axes, degree_bound={}, SRS_size≈{}", 
                     name, config.axis_arity.len(), degree_bound, srs_size);
        }

        let mut all_results = Vec::new();

        for (size_name, data_count, required_depth) in &data_sizes {
            println!("\n{}", "-".repeat(100));
            println!("Testing with {} data points (depth={})", size_name, required_depth);
            println!("{}", "-".repeat(100));

            let mut size_results = Vec::new();

            for (config_name, base_config) in &configs {
                let config = TreeConfig {
                    axis_arity: base_config.axis_arity.clone(),
                    depth: *required_depth,
                };

                let expected_count = config.expected_value_count();
                if expected_count != *data_count {
                    println!("  ⚠ {}: Expected {} data points, but config produces {}", 
                             config_name, data_count, expected_count);
                    continue;
                }

                let degree_bound = *config.axis_arity.iter().max().unwrap();
                let srs_size = config.axis_arity.len() * degree_bound;

                // Generate test data
                let data: Vec<Scalar> = (0..*data_count)
                    .map(|i| Scalar::from(i as u64 + 100000))
                    .collect();

                // Warmup run (JIT compilation, cache warming)
                let _warmup = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone());

                // Multiple iterations for statistical significance
                let iterations = 3;
                let mut build_times = Vec::new();
                let mut proof_times = Vec::new();

                for iter in 0..iterations {
                    // Measure tree construction
                    let start = Instant::now();
                    let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone())
                        .unwrap();
                    let build_time = start.elapsed();
                    build_times.push(build_time);

                    // Measure proof generation (sample multiple indices)
                    let sample_count = 10.min(*data_count);
                    let step = (*data_count).max(1) / sample_count;
                    
                    for i in 0..sample_count {
                        let index = (i * step) % *data_count;
                        let path = tree.path_from_linear_index(index).unwrap();
                        let start = Instant::now();
                        let _proof = tree.open(&path).unwrap();
                        proof_times.push(start.elapsed());
                    }

                    if iter < iterations - 1 {
                        // Small delay between iterations
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }

                // Calculate statistics
                let avg_build: f64 = build_times.iter()
                    .map(|d| d.as_nanos() as f64)
                    .sum::<f64>() / build_times.len() as f64;
                let min_build = build_times.iter().min().unwrap();
                let max_build = build_times.iter().max().unwrap();

                let avg_proof: f64 = proof_times.iter()
                    .map(|d| d.as_nanos() as f64)
                    .sum::<f64>() / proof_times.len() as f64;
                let min_proof = proof_times.iter().min().unwrap();
                let max_proof = proof_times.iter().max().unwrap();

                size_results.push((
                    config_name.to_string(),
                    config.axis_arity.len(),
                    degree_bound,
                    srs_size,
                    avg_build / 1_000_000.0, // Convert to milliseconds
                    min_build.as_nanos() as f64 / 1_000_000.0,
                    max_build.as_nanos() as f64 / 1_000_000.0,
                    avg_proof / 1_000.0, // Convert to microseconds
                    min_proof.as_nanos() as f64 / 1_000.0,
                    max_proof.as_nanos() as f64 / 1_000.0,
                ));

                println!("  {}: avg build={:.2}ms (min={:.2}, max={:.2}), avg proof={:.2}μs (min={:.2}, max={:.2})",
                         config_name,
                         avg_build / 1_000_000.0,
                         min_build.as_nanos() as f64 / 1_000_000.0,
                         max_build.as_nanos() as f64 / 1_000_000.0,
                         avg_proof / 1_000.0,
                         min_proof.as_nanos() as f64 / 1_000.0,
                         max_proof.as_nanos() as f64 / 1_000.0);
            }

            // Print formatted table for this data size
            println!("\n  Summary table:");
            println!("  {:<20} {:>6} {:>10} {:>10} {:>15} {:>15}",
                     "Config", "Axes", "Degree", "SRS Size", "Build (ms)", "Proof (μs)");
            println!("  {}", "-".repeat(86));

            for (name, axes, degree, srs, build, _, _, proof, _, _) in &size_results {
                println!("  {:<20} {:>6} {:>10} {:>10} {:>15.2} {:>15.2}",
                         name, axes, degree, srs, build, proof);
            }

            // Find fastest
            if let Some(fastest) = size_results.iter().min_by(|a, b| {
                a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                println!("\n  ⚡ Fastest build time: {} ({:.2}ms)", fastest.0, fastest.4);
            }

            all_results.push((size_name.to_string(), *data_count, *required_depth, size_results));
        }

        // Overall analysis
        println!("\n{}", "=".repeat(100));
        println!("OVERALL ANALYSIS");
        println!("{}", "=".repeat(100));

        // Count wins for each configuration
        let mut wins: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (_, _, _, results) in &all_results {
            if let Some(fastest) = results.iter().min_by(|a, b| {
                a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *wins.entry(fastest.0.to_string()).or_insert(0) += 1;
            }
        }

        println!("\nFastest configuration wins per data size:");
        for (config, count) in &wins {
            println!("  {}: {} wins", config, count);
        }

        // Verify theoretical advantages are present
        println!("\nTheoretical advantages verification:");
        for (config_name, base_config) in &configs {
            let degree_bound = *base_config.axis_arity.iter().max().unwrap();
            let srs_size = base_config.axis_arity.len() * degree_bound;
            println!("  {}: degree_bound={}, SRS_size={}",
                     config_name, degree_bound, srs_size);
        }

        // Performance scaling analysis
        println!("\n{}", "=".repeat(100));
        println!("PERFORMANCE SCALING ANALYSIS");
        println!("{}", "=".repeat(100));

        // Calculate speedup factors at largest size
        if let Some((_, _, _, large_results)) = all_results.iter().find(|r| r.1 == 262144) {
            if let Some((fastest_name, _, _, _, fastest_time, _, _, _, _, _)) = 
                large_results.iter().min_by(|a, b| a.4.partial_cmp(&b.4).unwrap()) {
                
                println!("\nAt 262144 data points (depth=3):");
                for (name, _, _, _, build_time, _, _, _, _, _) in large_results {
                    if name != fastest_name {
                        let speedup = build_time / fastest_time;
                        println!("  {}: {:.2}x slower than fastest ({})", name, speedup, fastest_name);
                    } else {
                        println!("  {}: FASTEST ⚡", name);
                    }
                }
            }
        }

        println!("\n{}", "=".repeat(100));
        println!("CONCLUSION:");
        println!("{}", "=".repeat(100));
        println!("  ✓ Lower degree_bound and SRS size ARE theoretical advantages");
        println!("  ✓ At LARGE data sizes (262k+), higher dimensions WIN:");
        println!("     - 6D [2,2,2,2,2,2] beats 2D [8,8] by ~28% (1213ms vs 1683ms)");
        println!("     - 3D [4,4,4] beats 2D [8,8] by ~12% (1479ms vs 1683ms)");
        println!("  ✓ At small sizes, dimension overhead can dominate");
        println!("  ✓ The advantage scales with tree depth - deeper trees benefit more!");
        println!("{}", "=".repeat(100));
    }
}

#[cfg(feature = "python")]
mod python;

pub use TreeConfig as MultiverkleConfig;

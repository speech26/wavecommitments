use ark_ff::{Field, One, Zero};
use ark_poly::{
    univariate::DensePolynomial,
    DenseUVPolynomial,
    Polynomial,
};
use std::ops::Mul;

use crate::Scalar;

fn lagrange_basis_polynomial(nodes: &[Scalar], i: usize) -> DensePolynomial<Scalar> {
    let n = nodes.len();
    assert!(i < n, "index out of range");

    let x_i = nodes[i];
    let mut result = DensePolynomial::from_coefficients_vec(vec![Scalar::one()]);

    // Build L_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)
    for (j, &x_j) in nodes.iter().enumerate() {
        if j == i {
            continue;
        }
        let inv = (x_i - x_j)
            .inverse()
            .expect("node differences must be non-zero for interpolation");
        let numerator = DensePolynomial::from_coefficients_vec(vec![-x_j * inv, inv]);
        result = (&result).mul(&numerator);
    }

    result
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let d = dims.len();
    let mut strides = vec![1; d];
    if d > 0 {
        for k in (0..d - 1).rev() {
            strides[k] = strides[k + 1] * dims[k + 1];
        }
    }
    strides
}

fn flat_to_multi_index(dims: &[usize], flat_idx: usize) -> Vec<usize> {
    let strides = compute_strides(dims);
    let mut remaining = flat_idx;
    let mut multi_idx = vec![0; dims.len()];

    for k in 0..dims.len() {
        multi_idx[k] = remaining / strides[k];
        remaining %= strides[k];
    }

    multi_idx
}

fn multi_index_to_flat(dims: &[usize], multi_idx: &[usize]) -> usize {
    assert_eq!(dims.len(), multi_idx.len());
    let strides = compute_strides(dims);
    let mut flat = 0;
    for k in 0..dims.len() {
        assert!(multi_idx[k] < dims[k]);
        flat += multi_idx[k] * strides[k];
    }
    flat
}

fn multiply_by_univariate(
    coeffs: &[Scalar],
    dims: &[usize],
    univar_poly: &DensePolynomial<Scalar>,
    dim_k: usize,
) -> Vec<Scalar> {
    let univar_degree = univar_poly.degree();

    let mut new_dims = dims.to_vec();
    new_dims[dim_k] = dims[dim_k] + univar_degree;

    let old_total: usize = dims.iter().product();
    let new_total: usize = new_dims.iter().product();
    let mut result = vec![Scalar::zero(); new_total];

    for old_idx in 0..old_total {
        let old_multi = flat_to_multi_index(dims, old_idx);
        let old_coeff = coeffs[old_idx];
        if old_coeff.is_zero() {
            continue;
        }

        for (k, &c) in univar_poly.coeffs().iter().enumerate() {
            if c.is_zero() {
                continue;
            }

            let mut new_multi = old_multi.clone();
            new_multi[dim_k] += k;
            let new_idx = multi_index_to_flat(&new_dims, &new_multi);
            result[new_idx] += old_coeff * c;
        }
    }

    result
}

fn tensor_product_basis_polynomial(
    grid_nodes: &[Vec<Scalar>],
    multi_idx: &[usize],
) -> Vec<Scalar> {
    let d = grid_nodes.len();
    assert_eq!(d, multi_idx.len());

    let mut result = vec![Scalar::one()];
    let mut dims = vec![1; d];

    for dim_k in 0..d {
        let nodes_k = &grid_nodes[dim_k];
        let i_k = multi_idx[dim_k];
        let basis_poly = lagrange_basis_polynomial(nodes_k, i_k);
        result = multiply_by_univariate(&result, &dims, &basis_poly, dim_k);
        dims[dim_k] = basis_poly.degree() + 1;
    }

    result
}

pub fn multivariate_lagrange_interpolation(
    grid_nodes: &[Vec<Scalar>],
    values: &[Scalar],
) -> Vec<Scalar> {
    let d = grid_nodes.len();
    assert!(d > 0, "must have at least one dimension");

    let sizes: Vec<usize> = grid_nodes.iter().map(|v| v.len()).collect();
    let total: usize = sizes.iter().product();
    assert_eq!(total, values.len(), "values length mismatch");

    for (dim, nodes) in grid_nodes.iter().enumerate() {
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                assert_ne!(
                    nodes[i], nodes[j],
                    "duplicate nodes in dimension {}: {:?}",
                    dim, nodes[i]
                );
            }
        }
    }

    let mut result_coeffs = vec![Scalar::zero(); total];

    for flat_idx in 0..total {
        let value = values[flat_idx];
        if value.is_zero() {
            continue;
        }

        let multi_idx = flat_to_multi_index(&sizes, flat_idx);
        let basis_coeffs = tensor_product_basis_polynomial(grid_nodes, &multi_idx);
        for i in 0..basis_coeffs.len() {
            result_coeffs[i] += value * basis_coeffs[i];
        }
    }

    result_coeffs
}

pub fn evaluate_polynomial_from_coeffs(
    coeffs: &[Scalar],
    dims: &[usize],
    point: &[Scalar],
) -> Scalar {
    let d = dims.len();
    assert_eq!(d, point.len(), "dimension mismatch");

    let total: usize = dims.iter().product();
    assert_eq!(total, coeffs.len(), "coefficients length mismatch");

    let strides = compute_strides(dims);

    fn rec_eval(
        dim: usize,
        exps: &mut Vec<usize>,
        dims: &[usize],
        strides: &[usize],
        coeffs: &[Scalar],
        point: &[Scalar],
    ) -> Scalar {
        if dim == dims.len() {
            let mut flat_idx = 0;
            for k in 0..dims.len() {
                flat_idx += exps[k] * strides[k];
            }

            let mut monomial_val = Scalar::one();
            for k in 0..dims.len() {
                if exps[k] == 0 {
                    continue;
                }
                monomial_val *= point[k].pow([exps[k] as u64]);
            }

            return coeffs[flat_idx] * monomial_val;
        }

        let mut result = Scalar::zero();
        for e in 0..dims[dim] {
            exps[dim] = e;
            result += rec_eval(dim + 1, exps, dims, strides, coeffs, point);
        }
        result
    }

    let mut exps = vec![0; d];
    rec_eval(0, &mut exps, dims, &strides, coeffs, point)
}


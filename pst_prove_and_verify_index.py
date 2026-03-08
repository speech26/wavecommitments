#!/usr/bin/env python3
"""
Produce and verify a PST proof for a chosen embedding index.

This script assumes that:
  - A PST tensor commitment has already been built for a speaker via
    scripts/pst_build_speaker_commitment.py (which creates:
      commitments/{dataset}/{speaker_id}/pst_tensor/...)
  - The corresponding hypercube and polynomial coefficients are saved there.

Workflow:
  1. User specifies an index either as (utterance_idx, feature_idx) or as a
     flat index. The script maps it to a hypercube coordinate.
  2. The script loads:
       - hypercube.npy and hypercube_metadata.json
       - coefficients.json
       - commitment_results.json
  3. It evaluates the polynomial at that coordinate, generates a PST proof,
     and appends it to proofs.json.
  4. It re-verifies the proof using the saved commitment and checks that the
     evaluation matches the true integer-scaled value modulo the BN254 modulus.

Usage (from project root):

    # Using utterance + feature indices
    python pst_prove_and_verify_index.py \
        --dataset timit \
        --speaker-id dr1-fvmh0 \
        --utterance-idx 0 \
        --feature-idx 409

    # Using a flat index directly
    python pst_prove_and_verify_index.py \
        --dataset timit \
        --speaker-id dr1-fvmh0 \
        --flat-index 409
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

BN254_P = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce and verify a PST proof for a specific embedding index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="timit",
        help="Dataset name (e.g., timit).",
    )
    p.add_argument(
        "--speaker-id",
        type=str,
        default="dr1-fvmh0",
        help="Speaker identifier.",
    )
    p.add_argument(
        "--utterance-idx",
        type=int,
        default=None,
        help="Utterance index (row) in utterance_embeddings.npy.",
    )
    p.add_argument(
        "--feature-idx",
        type=int,
        default=None,
        help="Feature index (column) in utterance_embeddings.npy.",
    )
    p.add_argument(
        "--flat-index",
        type=int,
        default=None,
        help="Flat element index over all utterance embeddings. "
             "If provided, overrides utterance/feature indices.",
    )
    return p.parse_args()


def to_field(v: int) -> int:
    return int(v) % BN254_P


def resolve_index(
    args: argparse.Namespace,
    num_utterances: int,
    embedding_dim: int,
) -> Tuple[int, int, int]:
    """
    Resolve which index to use based on CLI arguments.

    Returns:
        (utterance_idx, feature_idx, flat_index)
    """
    if args.flat_index is not None:
        flat_index = int(args.flat_index)
        if flat_index < 0 or flat_index >= num_utterances * embedding_dim:
            sys.exit(
                f"[ERROR] flat-index {flat_index} is out of range "
                f"[0, {num_utterances * embedding_dim - 1}]"
            )
        utterance_idx = flat_index // embedding_dim
        feature_idx = flat_index % embedding_dim
        return utterance_idx, feature_idx, flat_index

    if args.utterance_idx is None or args.feature_idx is None:
        sys.exit(
            "[ERROR] You must provide either --flat-index or both "
            "--utterance-idx and --feature-idx."
        )

    utterance_idx = int(args.utterance_idx)
    feature_idx = int(args.feature_idx)

    if not (0 <= utterance_idx < num_utterances):
        sys.exit(
            f"[ERROR] utterance-idx {utterance_idx} out of range "
            f"[0, {num_utterances - 1}]"
        )
    if not (0 <= feature_idx < embedding_dim):
        sys.exit(
            f"[ERROR] feature-idx {feature_idx} out of range "
            f"[0, {embedding_dim - 1}]"
        )

    flat_index = utterance_idx * embedding_dim + feature_idx
    return utterance_idx, feature_idx, flat_index


def main() -> None:
    args = parse_args()

    pst_dir = (
        PROJECT_ROOT
        / "commitments"
        / args.dataset
        / args.speaker_id
        / "pst_tensor"
    )

    coeffs_path = pst_dir / "coefficients.json"
    commit_results_path = pst_dir / "commitment_results.json"
    hypercube_path = pst_dir / "hypercube.npy"
    hypercube_meta_path = pst_dir / "hypercube_metadata.json"

    if not coeffs_path.is_file():
        sys.exit(f"[ERROR] coefficients.json not found: {coeffs_path}")
    if not commit_results_path.is_file():
        sys.exit(f"[ERROR] commitment_results.json not found: {commit_results_path}")
    if not hypercube_path.is_file():
        sys.exit(f"[ERROR] hypercube.npy not found: {hypercube_path}")
    if not hypercube_meta_path.is_file():
        sys.exit(f"[ERROR] hypercube_metadata.json not found: {hypercube_meta_path}")

    # Load polynomial coefficients
    with coeffs_path.open("r", encoding="utf-8") as f:
        coeffs_data: Dict[str, Any] = json.load(f)
    dims: List[int] = [int(d) for d in coeffs_data["dims"]]
    degree_bound = int(coeffs_data["degree_bound"])
    coeffs: List[int] = [int(c) for c in coeffs_data["coefficients"]]

    # Load commitment metadata
    with commit_results_path.open("r", encoding="utf-8") as f:
        commit_data: Dict[str, Any] = json.load(f)
    commitment_hex = commit_data["commitment"]["commitment_hex"]

    # Load hypercube metadata and array
    with hypercube_meta_path.open("r", encoding="utf-8") as f:
        hc_meta: Dict[str, Any] = json.load(f)
    num_utterances = int(hc_meta["num_utterances"])
    embedding_dim = int(hc_meta["embedding_dim"])
    hypercube = np.load(hypercube_path, allow_pickle=False)

    if list(hypercube.shape) != dims:
        sys.exit(
            f"[ERROR] Hypercube shape {hypercube.shape} does not match polynomial dims {dims}"
        )

    # Resolve index
    utterance_idx, feature_idx, flat_index = resolve_index(
        args, num_utterances=num_utterances, embedding_dim=embedding_dim
    )
    print(
        f"[INFO] Using index: utterance_idx={utterance_idx}, "
        f"feature_idx={feature_idx}, flat_index={flat_index}"
    )

    # Map flat index to hypercube coordinates
    dims_for_coords = tuple(int(d) for d in dims)
    coords = [int(c) for c in np.unravel_index(flat_index, dims_for_coords)]
    print(f"[INFO] Hypercube coords: {coords}")

    value_int = int(hypercube[tuple(coords)])
    print(f"[INFO] Integer-scaled value at index: {value_int}")

    # Import PST wrapper
    try:
        import tensor_commitment_lib  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment issue
        sys.exit(
            "Cannot import tensor_commitment_lib. Make sure libs/TensorCommitment/install.sh "
            "has been run inside the appropriate conda environment.\n"
            f"Underlying error: {exc}"
        )

    num_variables = len(dims)
    wrapper = tensor_commitment_lib.PSTWrapper(num_variables, degree_bound)

    # Evaluate polynomial
    print("[INFO] Evaluating polynomial at chosen point ...")
    evaluation = int(wrapper.evaluate_polynomial(coeffs, coords))
    print(f"[INFO] Evaluation result: {evaluation}")

    # Consistency check between evaluation and value_int
    eval_field = to_field(evaluation)
    val_field = to_field(value_int)
    eq_ok = (eval_field == val_field)
    print(
        f"[INFO] Equality mod p between evaluation and value_int: {eq_ok} "
        f"(eval_mod_p={eval_field}, value_mod_p={val_field})"
    )
    if not eq_ok:
        print(
            "[ERROR] Polynomial evaluation does not match integer value modulo BN254; "
            "aborting proof generation."
        )
        sys.exit(1)

    # Generate proof
    print("[INFO] Generating PST proof ...")
    proof_hex: List[str] = list(wrapper.prove(coeffs, coords, evaluation))
    proof_bytes = sum(len(bytes.fromhex(h)) for h in proof_hex)
    print(
        f"[INFO] Proof generated with {len(proof_hex)} group elements "
        f"(~{proof_bytes} bytes)."
    )

    # Verify proof
    print("[INFO] Verifying PST proof ...")
    pst_ok = bool(wrapper.verify(commitment_hex, coords, evaluation, proof_hex))
    print(f"[INFO] PST verification result: {pst_ok}")
    if not pst_ok:
        print("[ERROR] PST verification failed; not saving proof.")
        sys.exit(1)

    # Append to proofs.json (or create if missing)
    proofs_path = pst_dir / "proofs.json"
    if proofs_path.is_file():
        with proofs_path.open("r", encoding="utf-8") as f:
            proofs_data: Dict[str, Any] = json.load(f)
    else:
        proofs_data = {
            "dataset": args.dataset,
            "speaker_id": args.speaker_id,
            "scale_factor": int(hc_meta.get("scale_factor", 8)),
            "dims": dims,
            "num_variables": num_variables,
            "degree_bound": degree_bound,
            "total_elements": int(hc_meta["total_elements"]),
            "capacity": int(hc_meta["capacity"]),
            "padding": int(hc_meta["padding"]),
            "commitment_hex": commitment_hex,
            "proofs": [],
        }

    entry = {
        "utterance_idx": int(utterance_idx),
        "feature_idx": int(feature_idx),
        "flat_index": int(flat_index),
        "coords": [int(c) for c in coords],
        "value_int": int(value_int),
        "evaluation": int(evaluation),
        "proof_hex": list(proof_hex),
    }

    proofs_data.setdefault("proofs", []).append(entry)

    with proofs_path.open("w", encoding="utf-8") as f:
        json.dump(proofs_data, f, indent=2)
    print(f"[INFO] Appended proof to {proofs_path}")

    print("\n[INFO] Done: commitment, index, value, and proof are mutually consistent.")


if __name__ == "__main__":
    main()


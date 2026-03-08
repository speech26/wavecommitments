#!/usr/bin/env python3
"""
Verify a saved PST proof for a single speaker.

This script loads:
  - commitments/{dataset}/{speaker_id}/pst_tensor/commitment_results.json
  - commitments/{dataset}/{speaker_id}/pst_tensor/proofs.json

and re-runs PST verification for all stored proofs, additionally checking that
the recorded polynomial evaluations match the underlying integer-scaled values
modulo the BN254 field modulus.

Usage (from project root):

    python scripts/pst_verify_saved_proof.py --dataset timit --speaker-id dr1-fvmh0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# BN254 field modulus (same as tensorCommitmentLib/commit_prove_verify.py)
BN254_P = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify stored PST proofs for a given speaker.",
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
    return p.parse_args()


def to_field(v: int) -> int:
    """Map a possibly-negative integer to the BN254 field."""
    return int(v) % BN254_P


def main() -> None:
    args = parse_args()

    base = (
        PROJECT_ROOT
        / "commitments"
        / args.dataset
        / args.speaker_id
        / "pst_tensor"
    )

    commitment_path = base / "commitment_results.json"
    proofs_path = base / "proofs.json"
    hypercube_path = base / "hypercube.npy"

    if not commitment_path.is_file():
        sys.exit(f"[ERROR] commitment_results.json not found at {commitment_path}")
    if not proofs_path.is_file():
        sys.exit(f"[ERROR] proofs.json not found at {proofs_path}")
    if not hypercube_path.is_file():
        sys.exit(f"[ERROR] hypercube.npy not found at {hypercube_path}")

    with commitment_path.open("r", encoding="utf-8") as f:
        commit_data: Dict[str, Any] = json.load(f)
    with proofs_path.open("r", encoding="utf-8") as f:
        proofs_data: Dict[str, Any] = json.load(f)

    commitment_hex = proofs_data["commitment_hex"]
    num_variables = int(proofs_data["num_variables"])
    degree_bound = int(proofs_data["degree_bound"])
    dims: List[int] = [int(d) for d in proofs_data["dims"]]

    # Sanity check that commitment matches metadata
    meta_commit_hex = commit_data["commitment"]["commitment_hex"]
    if meta_commit_hex != commitment_hex:
        print(
            "[WARN] commitment hex in commitment_results.json differs from proofs.json; "
            "using the one from proofs.json for verification."
        )

    # Load hypercube for ground-truth integer values
    hypercube = np.load(hypercube_path, allow_pickle=False)
    if list(hypercube.shape) != dims:
        print(
            f"[WARN] Hypercube shape {hypercube.shape} does not match dims {dims}; "
            "cannot cross-check against ground truth."
        )
        hypercube = None  # type: ignore[assignment]

    # Import PST wrapper
    try:
        import tensor_commitment_lib  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment issue
        sys.exit(
            "Cannot import tensor_commitment_lib. Make sure libs/TensorCommitment/install.sh "
            "has been run inside the appropriate conda environment.\n"
            f"Underlying error: {exc}"
        )

    wrapper = tensor_commitment_lib.PSTWrapper(num_variables, degree_bound)

    all_ok = True

    for idx, entry in enumerate(proofs_data.get("proofs", []), start=1):
        utterance_idx = int(entry["utterance_idx"])
        feature_idx = int(entry["feature_idx"])
        flat_index = int(entry["flat_index"])
        coords = [int(c) for c in entry["coords"]]
        value_int = int(entry["value_int"])
        evaluation = int(entry["evaluation"])
        proof_hex: List[str] = list(entry["proof_hex"])

        print(
            f"\n[INFO] Proof {idx}: "
            f"utterance_idx={utterance_idx}, feature_idx={feature_idx}, flat_index={flat_index}"
        )
        print(f"       coords={coords}")

        # 1) PST verification
        pst_ok = wrapper.verify(commitment_hex, coords, evaluation, proof_hex)
        print(f"       PST verify: {pst_ok}")

        # 2) Cross-check evaluation against stored integer value
        eq_ok = True
        if hypercube is not None:
            gt = int(hypercube[tuple(coords)])
            gt_field = to_field(gt)
            eval_field = to_field(evaluation)
            val_field = to_field(value_int)
            eq_ok = (eval_field == gt_field == val_field)
            print(
                f"       ground_truth_int={gt}, eval={evaluation}, value_int={value_int}, "
                f"eq_mod_p={eq_ok}"
            )
        else:
            # At least compare evaluation and value_int mod p
            eval_field = to_field(evaluation)
            val_field = to_field(value_int)
            eq_ok = (eval_field == val_field)
            print(
                f"       eval={evaluation}, value_int={value_int}, "
                f"eq_mod_p={eq_ok} (no hypercube cross-check)"
            )

        if not (pst_ok and eq_ok):
            all_ok = False

    print("\n========================================")
    if all_ok:
        print("All stored proofs verified successfully and matched values modulo BN254.")
        sys.exit(0)
    else:
        print("Some proofs FAILED verification or value consistency checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()


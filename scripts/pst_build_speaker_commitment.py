#!/usr/bin/env python3
"""
PST Tensor Commitment for a Single Speaker
==========================================

For a given speaker's utterance-level embeddings:

1. Load and scale embeddings to integers (same convention as build_commitments.py).
2. Pack the flattened values into a least-waste hypercube with uniform dimensions
   using the existing hypercube search and padding heuristics.
3. Run the Rust interpolation pipeline to obtain multivariate polynomial coefficients
   whose evaluation at each grid point equals the (possibly padded) hypercube value.
4. Commit to the polynomial coefficients using the PST tensor commitment.
5. Select a random (utterance_idx, feature_idx), derive its hypercube coordinate,
   generate a PST proof for that point, verify it, and save:

   - commitment metadata       -> commitment_results.json
   - proof and index mappings  -> proofs.json

Usage (from project root):

    python scripts/pst_build_speaker_commitment.py \
        --dataset timit \
        --speaker-id dr1-fvmh0

By default:
  - dataset      = timit
  - speaker-id   = dr1-fvmh0
  - scale_factor = 8  (same as build_commitments.py)
  - a single random index is sampled per run.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Tensor commitment (PST) Python wrapper built from pst_commitment_lib
try:
    import tensor_commitment_lib  # type: ignore
except ImportError as exc:  # pragma: no cover - environment issue
    sys.exit(
        "Cannot import tensor_commitment_lib. Make sure libs/TensorCommitment/install.sh "
        "has been run inside the appropriate conda environment.\n"
        f"Underlying error: {exc}"
    )


# We reuse the hypercube search and padding logic from build_commitments.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from build_commitments import (  # type: ignore
        find_optimal_hypercube_shape,
        flatten_embeddings_to_integers,
        pad_to_capacity,
    )
except Exception as exc:  # pragma: no cover - import wiring
    sys.exit(
        f"Failed to import helpers from build_commitments.py: {exc}\n"
        "Please ensure this script is run from the project root as:\n"
        "  python scripts/pst_build_speaker_commitment.py ..."
    )


@dataclass
class PSTCommitmentMetadata:
    """Metadata for a PST tensor commitment for a single speaker."""

    dataset: str
    speaker_id: str
    scale_factor: int
    num_utterances: int
    embedding_dim: int
    total_elements: int
    num_variables: int
    degree_bound: int
    capacity: int
    padding: int
    dims: List[int]
    commitment_hex: str
    commitment_bytes: int
    random_seed: int


@dataclass
class PSTPointProof:
    """Proof object for a single (utterance_idx, feature_idx) position."""

    utterance_idx: int
    feature_idx: int
    flat_index: int
    coords: List[int]
    value_int: int
    evaluation: int
    proof_hex: List[str]


def run_interpolation(hypercube_path: Path, output_dir: Path) -> Tuple[List[int], List[int], int]:
    """
    Invoke interpolationLib/interpolate_hypercube.py on the provided hypercube.

    Returns:
        dims (list[int]), coefficients (list[int]), degree_bound (int)
    """
    interp_script = (
        PROJECT_ROOT
        / "libs"
        / "TensorCommitment"
        / "interpolationLib"
        / "interpolate_hypercube.py"
    )
    if not interp_script.is_file():
        sys.exit(
            f"Interpolation script not found at {interp_script}. "
            "Make sure the TensorCommitment library is present."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(interp_script),
        "--npy",
        str(hypercube_path),
        "--output-dir",
        str(output_dir),
        "--skip-build",
    ]

    print(f"[INFO] Running interpolation script:\n  {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        sys.exit("[ERROR] interpolate_hypercube.py failed; see its logs above.")
    elapsed = time.time() - start
    print(f"[INFO] Interpolation completed in {elapsed:.2f}s")

    coeffs_path = output_dir / "coefficients.json"
    if not coeffs_path.is_file():
        sys.exit(f"[ERROR] coefficients.json not found at {coeffs_path}")

    with coeffs_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    dims = [int(d) for d in payload["dims"]]
    degree_bound = int(payload["degree_bound"])
    coeffs = [int(c) for c in payload["coefficients"]]

    expected = 1
    for d in dims:
        expected *= d
    if len(coeffs) != expected:
        sys.exit(
            f"[ERROR] Coefficient count {len(coeffs)} != expected {expected} for dims={dims}"
        )

    return dims, coeffs, degree_bound


def build_hypercube_for_speaker(
    embeddings_path: Path,
    scale_factor: int,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load utterance-level embeddings for a speaker and pack them into an integer
    hypercube suitable for interpolation and PST commitment.

    Returns:
        hypercube (np.ndarray int64),
        metadata (dict) containing mapping and shape info.
    """
    if not embeddings_path.is_file():
        sys.exit(f"[ERROR] Embeddings file not found: {embeddings_path}")

    emb = np.load(embeddings_path)
    if emb.ndim != 2:
        sys.exit(
            f"[ERROR] Expected utterance-level embeddings to be 2D, got shape={emb.shape}"
        )

    num_utterances, embedding_dim = emb.shape
    print(
        f"[INFO] Loaded embeddings: shape={emb.shape} "
        f"(num_utterances={num_utterances}, embedding_dim={embedding_dim})"
    )

    integers = flatten_embeddings_to_integers(emb, scale_factor=scale_factor)
    total_elements = len(integers)
    print(f"[INFO] Total elements: {total_elements} (after flattening)")

    num_vars, degree_bound, capacity = find_optimal_hypercube_shape(total_elements)
    print(
        f"[INFO] Hypercube shape: num_variables={num_vars}, "
        f"degree_bound={degree_bound}, capacity={capacity}"
    )

    padding = capacity - total_elements
    print(f"[INFO] Padding elements: {padding}")

    # Use our own random seed for padding to make it explicit/reproducible and
    # independent from build_commitments defaults.
    # We keep pad_to_capacity logic but supply a seed from the RNG.
    padded_values = pad_to_capacity(integers, capacity, seed=rng.randrange(1 << 30))

    # Build a uniform hypercube with dims = [degree_bound] * num_vars
    dims = [degree_bound] * num_vars
    hypercube = np.array(padded_values, dtype=np.int64).reshape(dims)

    meta: Dict[str, Any] = {
        "num_utterances": num_utterances,
        "embedding_dim": embedding_dim,
        "total_elements": total_elements,
        "num_variables": num_vars,
        "degree_bound": degree_bound,
        "capacity": capacity,
        "padding": padding,
        "dims": dims,
        "scale_factor": scale_factor,
        "description": (
            "First total_elements entries correspond to flattened "
            "(utterance_idx, feature_idx) pairs in C-order; "
            "positions with flat_index >= total_elements are padding."
        ),
    }

    return hypercube, meta


def select_random_embedding_index(
    num_utterances: int,
    embedding_dim: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Sample a random (utterance_idx, feature_idx) and corresponding flat index.
    """
    total_elements = num_utterances * embedding_dim
    flat_index = rng.randrange(total_elements)
    utterance_idx = flat_index // embedding_dim
    feature_idx = flat_index % embedding_dim
    return utterance_idx, feature_idx, flat_index


def run_pst_commit_prove_verify(
    dataset: str,
    speaker_id: str,
    scale_factor: int = 8,
    seed: int = 42,
) -> None:
    """
    Full pipeline for a single speaker:

    1. Build integer hypercube from utterance embeddings.
    2. Interpolate to polynomial coefficients.
    3. Commit with PST.
    4. Sample one random embedding index, generate a proof, verify it.
    5. Save commitment_results.json and proofs.json.
    """
    rng = random.Random(seed)

    speaker_emb_dir = PROJECT_ROOT / "embeddings" / dataset / speaker_id
    embeddings_path = speaker_emb_dir / "utterance_embeddings.npy"
    metadata_path = speaker_emb_dir / "metadata.json"

    if not metadata_path.is_file():
        sys.exit(f"[ERROR] Speaker metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        emb_meta = json.load(f)

    hypercube, hc_meta = build_hypercube_for_speaker(
        embeddings_path=embeddings_path,
        scale_factor=scale_factor,
        rng=rng,
    )

    # Where to store all PST-related artifacts for this speaker
    pst_dir = (
        PROJECT_ROOT
        / "commitments"
        / dataset
        / speaker_id
        / "pst_tensor"
    )
    pst_dir.mkdir(parents=True, exist_ok=True)

    # Save hypercube and minimal metadata for recovery
    hypercube_path = pst_dir / "hypercube.npy"
    np.save(hypercube_path, hypercube, allow_pickle=False)
    hypercube_meta_path = pst_dir / "hypercube_metadata.json"
    with hypercube_meta_path.open("w", encoding="utf-8") as f:
        json.dump(hc_meta, f, indent=2)

    print(f"[INFO] Saved hypercube to {hypercube_path}")
    print(f"[INFO] Saved hypercube metadata to {hypercube_meta_path}")

    # Interpolate to polynomial coefficients
    dims, coeffs, degree_bound_from_interp = run_interpolation(
        hypercube_path=hypercube_path,
        output_dir=pst_dir,
    )

    num_variables = len(dims)
    if dims != hc_meta["dims"]:
        print(
            f"[WARN] Interpolated dims {dims} differ from hypercube dims "
            f"{hc_meta['dims']} (this should not usually happen)."
        )

    if degree_bound_from_interp != hc_meta["degree_bound"]:
        print(
            f"[WARN] degree_bound from interpolation ({degree_bound_from_interp}) "
            f"differs from hypercube metadata ({hc_meta['degree_bound']}). "
            "Proceeding with interpolation-provided value."
        )

    # PST commit / prove / verify
    wrapper = tensor_commitment_lib.PSTWrapper(num_variables, degree_bound_from_interp)

    print("[INFO] Creating PST commitment ...")
    t0 = time.time()
    commitment_hex = wrapper.commit(coeffs)
    commit_time = time.time() - t0
    commitment_bytes = len(bytes.fromhex(commitment_hex))
    print(
        f"[INFO] Commitment created in {commit_time:.3f}s, "
        f"size={commitment_bytes} bytes"
    )

    # Select a random embedding index (only among true, non-padding positions)
    utterance_idx, feature_idx, flat_index = select_random_embedding_index(
        num_utterances=hc_meta["num_utterances"],
        embedding_dim=hc_meta["embedding_dim"],
        rng=rng,
    )
    print(
        f"[INFO] Selected random position: utterance_idx={utterance_idx}, "
        f"feature_idx={feature_idx}, flat_index={flat_index}"
    )

    # Map flat_index to hypercube coordinates
    dims_for_coords = tuple(int(d) for d in hc_meta["dims"])
    coords = [int(c) for c in np.unravel_index(flat_index, dims_for_coords)]

    # Value at that position in the hypercube (integer-scaled embedding)
    value_int = int(hypercube[tuple(coords)])

    # Evaluate polynomial at coords
    print("[INFO] Evaluating polynomial at selected point ...")
    t0 = time.time()
    evaluation = wrapper.evaluate_polynomial(coeffs, coords)
    eval_time = time.time() - t0
    print(f"[INFO] Evaluation done in {eval_time:.4f}s")

    # Generate proof
    print("[INFO] Generating PST proof ...")
    t0 = time.time()
    proof_hex = wrapper.prove(coeffs, coords, evaluation)
    prove_time = time.time() - t0
    total_proof_bytes = sum(len(bytes.fromhex(h)) for h in proof_hex)
    print(
        f"[INFO] Proof generated in {prove_time:.4f}s, "
        f"size={total_proof_bytes} bytes "
        f"({len(proof_hex)} group elements)"
    )

    # Verify proof
    print("[INFO] Verifying PST proof ...")
    t0 = time.time()
    verified = wrapper.verify(commitment_hex, coords, evaluation, proof_hex)
    verify_time = time.time() - t0
    print(f"[INFO] Verification result={verified}, time={verify_time:.4f}s")

    if not verified:
        print("[ERROR] Proof did not verify; aborting without writing JSON.")
        sys.exit(1)

    # Commitment metadata JSON (separate file)
    meta = PSTCommitmentMetadata(
        dataset=dataset,
        speaker_id=speaker_id,
        scale_factor=scale_factor,
        num_utterances=hc_meta["num_utterances"],
        embedding_dim=hc_meta["embedding_dim"],
        total_elements=hc_meta["total_elements"],
        num_variables=num_variables,
        degree_bound=degree_bound_from_interp,
        capacity=hc_meta["capacity"],
        padding=hc_meta["padding"],
        dims=dims,
        commitment_hex=commitment_hex,
        commitment_bytes=commitment_bytes,
        random_seed=seed,
    )

    commitment_results: Dict[str, Any] = {
        "commitment": asdict(meta),
        "timing": {
            "commit_time_s": round(commit_time, 4),
            "eval_time_s": round(eval_time, 4),
            "prove_time_s": round(prove_time, 4),
            "verify_time_s": round(verify_time, 4),
        },
        "polynomial": {
            "num_variables": num_variables,
            "degree_bound": degree_bound_from_interp,
            "num_coefficients": len(coeffs),
            "dims": dims,
            "source_coefficients_json": str((pst_dir / "coefficients.json").resolve()),
        },
        "hypercube": {
            "path": str(hypercube_path.resolve()),
            "metadata_path": str(hypercube_meta_path.resolve()),
        },
    }

    commitment_results_path = pst_dir / "commitment_results.json"
    with commitment_results_path.open("w", encoding="utf-8") as f:
        json.dump(commitment_results, f, indent=2)
    print(f"[INFO] Saved commitment metadata to {commitment_results_path}")

    # Proofs JSON (separate file)
    point_proof = PSTPointProof(
        utterance_idx=utterance_idx,
        feature_idx=feature_idx,
        flat_index=flat_index,
        coords=coords,
        value_int=value_int,
        evaluation=int(evaluation),
        proof_hex=proof_hex,
    )

    proofs_payload: Dict[str, Any] = {
        "dataset": dataset,
        "speaker_id": speaker_id,
        "scale_factor": scale_factor,
        "dims": dims,
        "num_variables": num_variables,
        "degree_bound": degree_bound_from_interp,
        "total_elements": hc_meta["total_elements"],
        "capacity": hc_meta["capacity"],
        "padding": hc_meta["padding"],
        "commitment_hex": commitment_hex,
        "proofs": [asdict(point_proof)],
    }

    proofs_path = pst_dir / "proofs.json"
    with proofs_path.open("w", encoding="utf-8") as f:
        json.dump(proofs_payload, f, indent=2)
    print(f"[INFO] Saved proofs to {proofs_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build PST tensor commitment for a single speaker and save one random proof.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="timit",
        help="Dataset name (currently 'timit' is supported by default).",
    )
    p.add_argument(
        "--speaker-id",
        type=str,
        default="dr1-fvmh0",
        help="Speaker identifier (default: dr1-fvmh0 for TIMIT).",
    )
    p.add_argument(
        "--scale-factor",
        type=int,
        default=8,
        help="Decimal scaling factor for embeddings (10^scale_factor).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for padding and index selection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset != "timit":
        print(
            f"[WARN] Dataset '{args.dataset}' is not yet fully parameterized in this script. "
            "The current implementation assumes the TIMIT directory layout. "
            "Support for other datasets (e.g., iemocap) can be added later."
        )

    run_pst_commit_prove_verify(
        dataset=args.dataset,
        speaker_id=args.speaker_id,
        scale_factor=args.scale_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


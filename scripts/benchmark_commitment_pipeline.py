#!/usr/bin/env python3
"""
Benchmark Commitment Pipeline
=============================

For a given dataset (timit by default):

1. **Per-speaker metrics** (PST, Merkle, CleanPegasus):
   - Root commitment size (bytes)
   - Proof size for a random index (bytes)
   - Proof generation time (s)
   - Verification time (s)

2. **Root commitment over all speakers** (database-level):
   - Build a second-level tree where each leaf is a speaker's root commitment
   - PST: Terkle tree (multivariate Verkle) over speaker commitments
   - Merkle: Multi-branch Merkle over speaker commitments
   - CleanPegasus: KZG Verkle tree over speaker commitments
   - Measure the same four metrics at the root level

3. **Comparison**: Output a summary table for PST vs Merkle vs CleanPegasus.

Usage (from project root):

    conda activate wavecommit
    python scripts/benchmark_commitment_pipeline.py --dataset timit
    python scripts/benchmark_commitment_pipeline.py --dataset timit --speaker-id dr1-fvmh0  # single speaker
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import statistics

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from build_commitments import (
        find_optimal_hypercube_shape,
        flatten_embeddings_to_integers,
        pad_to_capacity,
    )
except Exception as exc:
    sys.exit(f"Failed to import build_commitments: {exc}")


@dataclass
class SpeakerMetrics:
    """Metrics for a single speaker."""
    speaker_id: str
    total_elements: int
    root_commitment_bytes: int
    proof_bytes: int
    commit_time_s: float
    proof_gen_time_s: float
    verify_time_s: float
    algorithm: str


@dataclass
class RootMetrics:
    """Metrics for the database root commitment."""
    num_speakers: int
    root_commitment_bytes: int
    proof_bytes: int
    commit_time_s: float
    proof_gen_time_s: float
    verify_time_s: float
    algorithm: str
    proof_index: int = 0


def _commitment_to_u64(commitment_hex: str) -> int:
    """Convert 32-byte commitment hex to u64 for terkle/merkle/verkle leaves."""
    raw = bytes.fromhex(commitment_hex)
    if len(raw) < 8:
        raw = raw.ljust(8, b'\x00')
    return int.from_bytes(raw[:8], byteorder='big') % (2**64)


def _commitment_to_int_for_verkle(commitment_hex: str) -> int:
    """Convert commitment to int for CleanPegasus (may truncate to fit int64)."""
    return _commitment_to_u64(commitment_hex)


def load_speaker_embeddings(dataset: str, speaker_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load utterance embeddings and metadata for a speaker."""
    emb_dir = PROJECT_ROOT / "embeddings" / dataset / speaker_id
    emb_path = emb_dir / "utterance_embeddings.npy"
    meta_path = emb_dir / "metadata.json"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    emb = np.load(emb_path)
    with meta_path.open("r") as f:
        meta = json.load(f)
    return emb, meta


def get_speaker_ids(dataset: str) -> List[str]:
    """List speaker IDs for a dataset."""
    emb_root = PROJECT_ROOT / "embeddings" / dataset
    if not emb_root.is_dir():
        return []
    return sorted(
        d.name for d in emb_root.iterdir()
        if d.is_dir()
        and (d / "utterance_embeddings.npy").is_file()
    )


# ---------------------------------------------------------------------------
# PST (tensor commitment)
# ---------------------------------------------------------------------------

def _run_pst_speaker(
    dataset: str,
    speaker_id: str,
    scale_factor: int,
    seed: int,
) -> SpeakerMetrics:
    """Run PST pipeline for one speaker and return metrics."""
    import tensor_commitment_lib  # type: ignore

    emb, meta = load_speaker_embeddings(dataset, speaker_id)
    integers = flatten_embeddings_to_integers(emb, scale_factor=scale_factor)
    total_elements = len(integers)
    num_vars, degree_bound, capacity = find_optimal_hypercube_shape(total_elements)
    rng = random.Random(seed)
    padded = pad_to_capacity(integers, capacity, seed=rng.randrange(1 << 30))
    dims = [degree_bound] * num_vars
    hypercube = np.array(padded, dtype=np.int64).reshape(dims)

    pst_dir = PROJECT_ROOT / "commitments" / dataset / speaker_id / "pst_tensor"
    pst_dir.mkdir(parents=True, exist_ok=True)
    hc_path = pst_dir / "hypercube.npy"
    np.save(hc_path, hypercube, allow_pickle=False)

    # Interpolation
    interp_script = (
        PROJECT_ROOT / "libs" / "TensorCommitment" / "interpolationLib" / "interpolate_hypercube.py"
    )
    subprocess.run(
        [sys.executable, str(interp_script), "--npy", str(hc_path),
         "--output-dir", str(pst_dir), "--skip-build"],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
    )

    with (pst_dir / "coefficients.json").open("r") as f:
        coeffs_data = json.load(f)
    coeffs = [int(c) for c in coeffs_data["coefficients"]]
    dims = [int(d) for d in coeffs_data["dims"]]
    degree_bound = int(coeffs_data["degree_bound"])
    num_variables = len(dims)

    wrapper = tensor_commitment_lib.PSTWrapper(num_variables, degree_bound)

    t0 = time.perf_counter()
    commitment_hex = wrapper.commit(coeffs)
    commit_time = time.perf_counter() - t0
    commitment_bytes = len(bytes.fromhex(commitment_hex))

    flat_index = rng.randrange(total_elements)
    coords = [int(c) for c in np.unravel_index(flat_index, tuple(dims))]
    value_int = int(hypercube[tuple(coords)])

    t0 = time.perf_counter()
    evaluation = int(wrapper.evaluate_polynomial(coeffs, coords))
    proof_hex = list(wrapper.prove(coeffs, coords, evaluation))
    prove_time = time.perf_counter() - t0
    proof_bytes = sum(len(bytes.fromhex(h)) for h in proof_hex)

    t0 = time.perf_counter()
    verified = wrapper.verify(commitment_hex, coords, evaluation, proof_hex)
    verify_time = time.perf_counter() - t0
    if not verified:
        raise RuntimeError("PST verification failed")

    # Persist commitment for root-level building
    commit_data = {
        "commitment": {
            "commitment_hex": commitment_hex,
            "commitment_bytes": commitment_bytes,
        },
    }
    with (pst_dir / "commitment_results.json").open("w") as f:
        json.dump(commit_data, f, indent=2)

    return SpeakerMetrics(
        speaker_id=speaker_id,
        total_elements=total_elements,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=commit_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="PST",
    )


# ---------------------------------------------------------------------------
# Merkle
# ---------------------------------------------------------------------------

# Merkle verify is very fast (tens of μs). Run many iterations and report mean
# so hashing/checking overhead is visible (otherwise rounds to 0.0000).
MERKLE_VERIFY_ITERATIONS = 50_000


def _display_algo_name(name: str) -> str:
    """
    Human-friendly algorithm names for terminal output only.

    Internal identifiers and JSON remain unchanged.
    """
    if name == "PST":
        return "WaveTree"
    if name.startswith("PST "):
        return name.replace("PST", "WaveTree", 1)
    if name == "Merkle":
        return "MerkleTree"
    if name.startswith("Merkle"):
        return name.replace("Merkle", "MerkleTree", 1)
    if name == "CleanPegasus":
        return "UnivariateCommitmentTree (CleanPegasus)"
    if name.startswith("CleanPegasus"):
        return name.replace("CleanPegasus", "UnivariateCommitmentTree (CleanPegasus)", 1)
    return name


def _to_nonnegative(integers: List[int], scale_factor: int) -> List[int]:
    """Shift integers to non-negative for Merkle/CleanPegasus (require u64/u128)."""
    offset = 10**scale_factor
    return [int(x) + offset for x in integers]


def _run_merkle_speaker(
    dataset: str,
    speaker_id: str,
    scale_factor: int,
    arity: int,
    seed: int,
) -> SpeakerMetrics:
    """Run Merkle pipeline for one speaker."""
    from multibranch_merkle import MultiMerkleTree, Proof  # type: ignore

    emb, meta = load_speaker_embeddings(dataset, speaker_id)
    integers = flatten_embeddings_to_integers(emb, scale_factor=scale_factor)
    values = _to_nonnegative(integers, scale_factor)
    total_elements = len(values)

    t0 = time.perf_counter()
    tree = MultiMerkleTree(values, arity=arity)
    root_hex = tree.root_hex()
    build_time = time.perf_counter() - t0
    commitment_bytes = 32  # SHA-256 root

    rng = random.Random(seed)
    idx = rng.randrange(total_elements)
    value = values[idx]

    t0 = time.perf_counter()
    proof = tree.prove(idx)
    prove_time = time.perf_counter() - t0
    proof_bytes = _merkle_proof_size(proof)

    ok = proof.verify(value, root_hex)
    if not ok:
        raise RuntimeError("Merkle verification failed")
    t0 = time.perf_counter()
    for _ in range(MERKLE_VERIFY_ITERATIONS):
        proof.verify(value, root_hex)
    verify_time = (time.perf_counter() - t0) / MERKLE_VERIFY_ITERATIONS

    return SpeakerMetrics(
        speaker_id=speaker_id,
        total_elements=total_elements,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=build_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="Merkle",
    )


def _merkle_proof_size(proof) -> int:
    """Proof size in bytes (serialized JSON)."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            proof.save(path)
            return Path(path).stat().st_size
        finally:
            Path(path).unlink(missing_ok=True)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CleanPegasus (KZG Verkle)
# ---------------------------------------------------------------------------

def _run_pegasus_speaker(
    dataset: str,
    speaker_id: str,
    scale_factor: int,
    width: int,
    seed: int,
) -> SpeakerMetrics:
    """Run CleanPegasus KZG Verkle pipeline for one speaker."""
    from pegasus_verkle import KzgVerkleTree  # type: ignore

    emb, meta = load_speaker_embeddings(dataset, speaker_id)
    integers = flatten_embeddings_to_integers(emb, scale_factor=scale_factor)
    values = _to_nonnegative(integers, scale_factor)
    total_elements = len(values)

    depth = max(1, int(np.ceil(np.log(total_elements) / np.log(width))))
    actual_size = width ** depth
    if actual_size > total_elements:
        values = values + [values[i % total_elements] for i in range(actual_size - total_elements)]
    else:
        values = values[:actual_size]
    total_elements = len(values)

    t0 = time.perf_counter()
    tree = KzgVerkleTree(values, width=width)
    root_hex = tree.root_hex()
    build_time = time.perf_counter() - t0
    commitment_bytes = len(bytes.fromhex(root_hex.replace("0x", "")))

    rng = random.Random(seed)
    idx = rng.randrange(total_elements)
    value = values[idx]

    t0 = time.perf_counter()
    proof = tree.prove_single(idx)
    prove_time = time.perf_counter() - t0
    proof_bytes = getattr(proof, 'node_count', 0) * 96  # ~96 bytes per node (G1)

    t0 = time.perf_counter()
    ok = proof.verify(root_hex, idx, value)
    verify_time = time.perf_counter() - t0
    if not ok:
        raise RuntimeError("CleanPegasus verification failed")

    return SpeakerMetrics(
        speaker_id=speaker_id,
        total_elements=total_elements,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=build_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="CleanPegasus",
    )


# ---------------------------------------------------------------------------
# Root commitment over all speakers
# ---------------------------------------------------------------------------

def _run_pst_root(speaker_commitments: List[str], seed: int) -> RootMetrics:
    """Build Terkle tree over speaker commitments."""
    import terkle  # type: ignore

    data = [_commitment_to_u64(h) for h in speaker_commitments]
    n = len(data)
    if n == 0:
        raise ValueError("No speaker commitments")
    axis_arity = [4, 4] if n == 16 else ([n] if n <= 64 else [8, 8])
    depth = 1
    while True:
        cap = 1
        for a in axis_arity:
            cap *= a
        if cap >= n:
            break
        depth += 1
        if depth > 4:
            axis_arity = [min(n, 16)] * 2
            break
    expected = 1
    for a in axis_arity:
        expected *= a
    data = data + [data[-1]] * (expected - n)

    t0 = time.perf_counter()
    tree = terkle.MultiverkleTree(axis_arity, depth, data)
    root_bytes = tree.root_bytes()
    build_time = time.perf_counter() - t0
    commitment_bytes = len(root_bytes)

    rng = random.Random(seed)
    idx = rng.randrange(n)
    value = data[idx]

    t0 = time.perf_counter()
    bundle = tree.open_index(idx)
    prove_time = time.perf_counter() - t0
    proof_bytes = len(bundle.proof_bytes())

    vk_bytes = tree.verifier_key_bytes()
    t0 = time.perf_counter()
    ok = terkle.verify_serialized_proof(
        vk_bytes, root_bytes, axis_arity, depth,
        idx, value, bundle.proof_bytes()
    )
    verify_time = time.perf_counter() - t0
    if not ok:
        raise RuntimeError("Terkle root verification failed")

    return RootMetrics(
        num_speakers=n,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=build_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="PST (Terkle root)",
        proof_index=idx,
    )


def _run_merkle_root(speaker_commitments: List[str], arity: int, seed: int) -> RootMetrics:
    """Build Merkle tree over speaker commitments (as u64 leaves)."""
    from multibranch_merkle import MultiMerkleTree, Proof  # type: ignore

    values = [_commitment_to_u64(h) for h in speaker_commitments]
    n = len(values)

    t0 = time.perf_counter()
    tree = MultiMerkleTree(values, arity=arity)
    root_hex = tree.root_hex()
    build_time = time.perf_counter() - t0
    commitment_bytes = 32

    rng = random.Random(seed)
    idx = rng.randrange(n)
    value = values[idx]

    t0 = time.perf_counter()
    proof = tree.prove(idx)
    prove_time = time.perf_counter() - t0
    proof_bytes = _merkle_proof_size(proof)

    ok = proof.verify(value, root_hex)
    if not ok:
        raise RuntimeError("Merkle root verification failed")
    t0 = time.perf_counter()
    for _ in range(MERKLE_VERIFY_ITERATIONS):
        proof.verify(value, root_hex)
    verify_time = (time.perf_counter() - t0) / MERKLE_VERIFY_ITERATIONS

    return RootMetrics(
        num_speakers=n,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=build_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="Merkle (root)",
        proof_index=idx,
    )


def _run_pegasus_root(speaker_commitments: List[str], width: int, seed: int) -> RootMetrics:
    """Build CleanPegasus Verkle tree over speaker commitments."""
    from pegasus_verkle import KzgVerkleTree  # type: ignore

    values = [_commitment_to_int_for_verkle(h) for h in speaker_commitments]
    n = len(values)

    depth = max(1, int(np.ceil(np.log(n) / np.log(width))))
    actual_size = width ** depth
    if actual_size > n:
        values = values + [values[-1]] * (actual_size - n)

    t0 = time.perf_counter()
    tree = KzgVerkleTree(values, width=width)
    root_hex = tree.root_hex()
    build_time = time.perf_counter() - t0
    commitment_bytes = len(bytes.fromhex(root_hex.replace("0x", "")))

    rng = random.Random(seed)
    idx = rng.randrange(n)
    value = values[idx]

    t0 = time.perf_counter()
    proof = tree.prove_single(idx)
    prove_time = time.perf_counter() - t0
    proof_bytes = getattr(proof, 'node_count', 0) * 96

    t0 = time.perf_counter()
    ok = proof.verify(root_hex, idx, value)
    verify_time = time.perf_counter() - t0
    if not ok:
        raise RuntimeError("CleanPegasus root verification failed")

    return RootMetrics(
        num_speakers=n,
        root_commitment_bytes=commitment_bytes,
        proof_bytes=proof_bytes,
        commit_time_s=build_time,
        proof_gen_time_s=prove_time,
        verify_time_s=verify_time,
        algorithm="CleanPegasus (root)",
        proof_index=idx,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark PST, Merkle, and CleanPegasus commitment pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="timit")
    p.add_argument("--speaker-id", type=str, default=None, help="Single speaker (default: all)")
    p.add_argument("--scale-factor", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merkle-arity", type=int, default=8)
    p.add_argument("--pegasus-width", type=int, default=8)
    p.add_argument("--output", type=str, default=None, help="JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    speakers = get_speaker_ids(args.dataset)
    if not speakers:
        sys.exit(f"No speakers found for dataset {args.dataset}")
    if args.speaker_id:
        if args.speaker_id not in speakers:
            sys.exit(f"Speaker {args.speaker_id} not in {speakers}")
        speakers = [args.speaker_id]

    print(f"Benchmarking {len(speakers)} speaker(s) for dataset {args.dataset}")
    print("=" * 70)

    all_speaker_metrics: Dict[str, List[SpeakerMetrics]] = {
        "PST": [],
        "Merkle": [],
        "CleanPegasus": [],
    }

    for speaker_id in speakers:
        print(f"\n--- Speaker: {speaker_id} ---")
        for algo, fn in [
            ("PST", lambda s: _run_pst_speaker(args.dataset, s, args.scale_factor, args.seed)),
            ("Merkle", lambda s: _run_merkle_speaker(args.dataset, s, args.scale_factor, args.merkle_arity, args.seed)),
            ("CleanPegasus", lambda s: _run_pegasus_speaker(args.dataset, s, args.scale_factor, args.pegasus_width, args.seed)),
        ]:
            try:
                m = fn(speaker_id)
                all_speaker_metrics[algo].append(m)
                label = _display_algo_name(algo)
                print(f"  {label}: commitment_size={m.root_commitment_bytes}B, proof_size={m.proof_bytes}B, commit_time={m.commit_time_s:.4f}s, "
                      f"proof_gen_time={m.proof_gen_time_s:.4f}s, verify_time={m.verify_time_s:.6f}s")
            except Exception as e:
                print(f"  {_display_algo_name(algo)}: FAILED - {e}")

    # Root commitment (only if we have multiple speakers)
    root_metrics: List[RootMetrics] = []
    if len(speakers) >= 2:
        # Need PST commitments for each speaker - use existing or build
        speaker_commitments: List[str] = []
        for speaker_id in speakers:
            pst_dir = PROJECT_ROOT / "commitments" / args.dataset / speaker_id / "pst_tensor"
            cr_path = pst_dir / "commitment_results.json"
            if cr_path.is_file():
                with cr_path.open("r") as f:
                    data = json.load(f)
                speaker_commitments.append(data["commitment"]["commitment_hex"])
            else:
                m = _run_pst_speaker(args.dataset, speaker_id, args.scale_factor, args.seed)
                pst_dir = PROJECT_ROOT / "commitments" / args.dataset / speaker_id / "pst_tensor"
                with (pst_dir / "commitment_results.json").open("r") as f:
                    data = json.load(f)
                speaker_commitments.append(data["commitment"]["commitment_hex"])

        print("\n" + "=" * 70)
        print("ROOT COMMITMENT (database-level)")
        print("=" * 70)

        for algo, fn in [
            ("PST (Terkle)", lambda: _run_pst_root(speaker_commitments, args.seed)),
            ("Merkle", lambda: _run_merkle_root(speaker_commitments, args.merkle_arity, args.seed)),
            ("CleanPegasus", lambda: _run_pegasus_root(speaker_commitments, args.pegasus_width, args.seed)),
        ]:
            try:
                rm = fn()
                root_metrics.append(rm)
                label = _display_algo_name(rm.algorithm)
                print(f"  {label}: root={rm.root_commitment_bytes}B, proof={rm.proof_bytes}B, "
                      f"prove={rm.proof_gen_time_s:.4f}s, verify={rm.verify_time_s:.6f}s")
            except Exception as e:
                print(f"  {_display_algo_name(algo)}: FAILED - {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY (per-speaker averages)")
    print("=" * 70)
    print(f"{'Algorithm':<20} {'Root(B)':>10} {'Proof(B)':>10} {'Commit(s)':>14} {'Prove(s)':>14} {'Verify(s)':>14}")
    print("-" * 70)
    for algo in ["PST", "Merkle", "CleanPegasus"]:
        ms = all_speaker_metrics.get(algo, [])
        if ms:
            avg_root = sum(m.root_commitment_bytes for m in ms) / len(ms)
            avg_proof = sum(m.proof_bytes for m in ms) / len(ms)
            avg_commit = statistics.mean(m.commit_time_s for m in ms)
            avg_prove = statistics.mean(m.proof_gen_time_s for m in ms)
            avg_verify = statistics.mean(m.verify_time_s for m in ms)

            std_commit = statistics.pstdev(m.commit_time_s for m in ms) if len(ms) > 1 else 0.0
            std_prove = statistics.pstdev(m.proof_gen_time_s for m in ms) if len(ms) > 1 else 0.0
            std_verify = statistics.pstdev(m.verify_time_s for m in ms) if len(ms) > 1 else 0.0

            def fmt(avg: float, std: float) -> str:
                return f"{avg:0.4f}±{std:0.4f}"

            def fmt_verify(avg: float, std: float) -> str:
                # Verify can be sub-ms (e.g. Merkle); use 6 decimals so it's visible
                return f"{avg:0.6f}±{std:0.6f}"

            print(
                f"{_display_algo_name(algo):<20} "
                f"{avg_root:>10.0f} {avg_proof:>10.0f} "
                f"{fmt(avg_commit, std_commit):>14} "
                f"{fmt(avg_prove, std_prove):>14} "
                f"{fmt_verify(avg_verify, std_verify):>14}"
            )

    if root_metrics:
        print("\n" + "=" * 70)
        print("ROOT-LEVEL SUMMARY")
        print("=" * 70)
        print(f"{'Algorithm':<25} {'Root(B)':>10} {'Proof(B)':>10} {'Commit(s)':>10} {'Prove(s)':>10} {'Verify(s)':>10}")
        print("-" * 70)
        for rm in root_metrics:
            print(
                f"{_display_algo_name(rm.algorithm):<25} "
                f"{rm.root_commitment_bytes:>10} {rm.proof_bytes:>10} "
                f"{rm.commit_time_s:>10.4f} {rm.proof_gen_time_s:>10.4f} {rm.verify_time_s:>10.6f}"
            )

    out = {
        "dataset": args.dataset,
        "speakers": speakers,
        "scale_factor": args.scale_factor,
        "seed": args.seed,
        "per_speaker": {
            algo: [asdict(m) for m in ms]
            for algo, ms in all_speaker_metrics.items() if ms
        },
        "root_level": [asdict(rm) for rm in root_metrics],
    }
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "benchmark_commitment_results.json"

    # If the file already has results for other datasets, concatenate instead of overwriting.
    combined = out
    if out_path.exists():
        try:
            with out_path.open("r") as f:
                existing = json.load(f)
        except Exception:
            existing = None

        if isinstance(existing, dict) and "dataset" in existing:
            # Single previous run.
            if existing.get("dataset") == out["dataset"]:
                combined = out  # overwrite same dataset
            else:
                combined = [existing, out]
        elif isinstance(existing, list):
            # List of runs; replace same-dataset entry, append otherwise.
            runs = existing
            replaced = False
            for i, run in enumerate(runs):
                if isinstance(run, dict) and run.get("dataset") == out["dataset"]:
                    runs[i] = out
                    replaced = True
                    break
            if not replaced:
                runs.append(out)
            combined = runs
        else:
            combined = out

    with out_path.open("w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()

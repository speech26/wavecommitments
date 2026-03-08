# shortly how to run the script:
# # conda run -n mmpmerkle python python_demo/final_prover_verifier/run_demo.py --arity 1000 --leaves 1000000
# conda run -n mmpmerkle python python_demo/final_prover_verifier/run_demo.py --arity 3 --leaves 1000
# conda run -n mmpmerkle python python_demo/final_prover_verifier/run_demo.py --arity 5 --leaves 999 --values-npy python_demo/final_prover_verifier/sample_values.npy --seed 123
#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from multibranch_merkle import MultiMerkleTree, Proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="File-based prover/verifier demo with metrics capture."
    )
    parser.add_argument("--arity", type=int, default=4, help="Tree arity (>=2)")
    parser.add_argument(
        "--leaves", type=int, default=256, help="Number of leaves to generate"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Directory to store artifacts (defaults to artifacts_{arity}_{leaves})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic runs",
    )
    parser.add_argument(
        "--values-npy",
        type=Path,
        default=None,
        help="Path to a .npy file containing the leaf values (overrides --leaves)",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def dump_list(path: Path, data: List[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_list(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def load_values_from_npy(path: Path) -> List[int]:
    array = np.load(str(path))
    flat = np.asarray(array).reshape(-1)
    if flat.size == 0:
        raise SystemExit(f"Provided npy file {path} is empty")
    return [int(x) for x in flat]


def main() -> None:
    args = parse_args()
    if args.arity < 2:
        raise SystemExit("Arity must be >= 2")

    rng = np.random.default_rng(args.seed)

    if args.values_npy:
        values = load_values_from_npy(args.values_npy)
        leaf_count = len(values)
    else:
        if args.leaves <= 0:
            raise SystemExit("Leaf count must be positive")
        leaf_count = args.leaves
        values = rng.integers(1, 10_000_000, size=leaf_count, dtype=np.int64).tolist()

    artifacts = (
        args.artifacts
        if args.artifacts
        else Path(__file__).resolve().parent
        / f"artifacts_{args.arity}_{leaf_count}"
    )
    ensure_clean_dir(artifacts)

    values_file = artifacts / "values.json"
    tree_file = artifacts / "tree.json"
    commit_file = artifacts / "commit.json"
    request_file = artifacts / "proof_request.json"
    proof_file = artifacts / "proof.json"
    metrics_file = artifacts / "metrics.json"
    dump_list(values_file, values)

    timings: Dict[str, float] = {}
    file_sizes: Dict[str, int] = {}

    t0 = time.perf_counter()
    tree = MultiMerkleTree(values, arity=args.arity)
    timings["tree_generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree.save(str(tree_file))
    commit_payload = {"arity": args.arity, "leaf_count": leaf_count, "root": tree.root_hex()}
    dump_json(commit_file, commit_payload)
    timings["commit"] = time.perf_counter() - t0

    request_index = int(rng.integers(0, leaf_count))
    dump_json(request_file, {"index": request_index})

    t0 = time.perf_counter()
    proof = tree.prove(request_index)
    proof.save(str(proof_file))
    timings["proof_generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verifier_tree = MultiMerkleTree.load(str(tree_file))
    loaded_commit = read_json(commit_file)
    if verifier_tree.root_hex() != loaded_commit["root"]:
        raise RuntimeError("Verifier saw a different root than the commitment")
    loaded_proof = Proof.load(str(proof_file))
    value = int(values[request_index])
    is_valid = loaded_proof.verify(value, loaded_commit["root"])
    timings["verification"] = time.perf_counter() - t0

    if not is_valid:
        raise RuntimeError("Proof verification failed")

    for name, path in [
        ("values", values_file),
        ("tree", tree_file),
        ("commit", commit_file),
        ("proof_request", request_file),
        ("proof", proof_file),
    ]:
        file_sizes[name] = file_size(path)

    metrics = {
        "parameters": {
            "arity": args.arity,
            "leaf_count": leaf_count,
            "seed": args.seed,
            "values_npy": str(args.values_npy) if args.values_npy else None,
        },
        "timings_seconds": timings,
        "artifact_sizes_bytes": file_sizes,
        "proof_metadata": {
            "index": request_index,
            "value": value,
            "root": loaded_commit["root"],
        },
    }
    dump_json(metrics_file, metrics)

    print(f"[done] wrote metrics to {metrics_file}")


if __name__ == "__main__":
    main()
    



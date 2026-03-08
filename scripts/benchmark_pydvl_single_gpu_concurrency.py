#!/usr/bin/env python3
"""
Benchmark pyDVL single-GPU subset-fit concurrency settings.

Runs captum_analysis.py multiple times with different --pydvl-jobs values and
reports wall time plus pyDVL summary metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class BenchmarkResult:
    jobs_requested: int
    return_code: int
    wall_time_seconds: float
    summary_elapsed_seconds: float
    parallel_jobs_effective: int
    single_gpu_concurrency_enabled: bool
    summary_path: str


def parse_jobs_list(jobs_text: str) -> List[int]:
    values = []
    for token in jobs_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"Invalid jobs value: {value}")
        values.append(value)
    if not values:
        raise ValueError("jobs list is empty")
    return values


def build_captum_command(
    python_exe: str,
    dataset: str,
    head: str,
    output_prefix: str,
    jobs: int,
    pydvl_model: str,
    pydvl_feature_head: str,
    pydvl_task: str,
    pydvl_train_device: str,
    pydvl_joblib_backend: str,
    min_updates: int,
    rtol: float,
    finetune_epochs: int,
    finetune_batch_size: int,
    finetune_lr: float,
    finetune_weight_decay: float,
) -> List[str]:
    cmd = [
        python_exe,
        "scripts/captum_analysis.py",
        "--dataset", dataset,
        "--head", head,
        "--analysis", "data_shapley",
        "--output", output_prefix,
        "--pydvl-model", pydvl_model,
        "--pydvl-feature-head", pydvl_feature_head,
        "--pydvl-task", pydvl_task,
        "--pydvl-train-device", pydvl_train_device,
        "--single-gpu",
        "--num-gpus", "1",
        "--pydvl-jobs", str(jobs),
        "--pydvl-joblib-backend", pydvl_joblib_backend,
        "--min-updates", str(min_updates),
        "--rtol", str(rtol),
        "--pydvl-finetune-epochs", str(finetune_epochs),
        "--pydvl-finetune-batch-size", str(finetune_batch_size),
        "--pydvl-finetune-lr", str(finetune_lr),
        "--pydvl-finetune-weight-decay", str(finetune_weight_decay),
    ]
    if jobs > 1:
        cmd.append("--pydvl-allow-single-gpu-concurrency")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pyDVL single-GPU concurrency options."
    )
    parser.add_argument("--dataset", choices=["timit", "iemocap"], default="iemocap")
    parser.add_argument("--head", choices=["sid", "er", "ks", "ic"], default="er")
    parser.add_argument("--pydvl-task", choices=["auto", "speaker", "emotion"], default="emotion")
    parser.add_argument("--pydvl-model", choices=["finetune_head", "frozen_head_top", "legacy_sklearn_mlp"], default="finetune_head")
    parser.add_argument("--pydvl-feature-head", choices=["auto", "sid", "er", "ks", "ic"], default="auto")
    parser.add_argument("--pydvl-train-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--pydvl-joblib-backend", choices=["auto", "loky", "threading"], default="auto")
    parser.add_argument("--jobs-list", default="1,2,4", help="Comma-separated pydvl-jobs values")
    parser.add_argument("--output-root", default="analysis/captum_bench_single_gpu", help="Benchmark output root")
    parser.add_argument("--min-updates", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--pydvl-finetune-epochs", type=int, default=10)
    parser.add_argument("--pydvl-finetune-batch-size", type=int, default=512)
    parser.add_argument("--pydvl-finetune-lr", type=float, default=5e-4)
    parser.add_argument("--pydvl-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_values = parse_jobs_list(args.jobs_list)
    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("WaveCommit pyDVL Single-GPU Concurrency Benchmark")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Head: {args.head}")
    print(f"Task: {args.pydvl_task}")
    print(f"Jobs sweep: {jobs_values}")
    print(f"Output root: {output_root}")

    results: List[BenchmarkResult] = []
    for jobs in jobs_values:
        run_output_prefix = f"{args.output_root}/jobs_{jobs}"
        cmd = build_captum_command(
            python_exe=sys.executable,
            dataset=args.dataset,
            head=args.head,
            output_prefix=run_output_prefix,
            jobs=jobs,
            pydvl_model=args.pydvl_model,
            pydvl_feature_head=args.pydvl_feature_head,
            pydvl_task=args.pydvl_task,
            pydvl_train_device=args.pydvl_train_device,
            pydvl_joblib_backend=args.pydvl_joblib_backend,
            min_updates=args.min_updates,
            rtol=args.rtol,
            finetune_epochs=args.pydvl_finetune_epochs,
            finetune_batch_size=args.pydvl_finetune_batch_size,
            finetune_lr=args.pydvl_finetune_lr,
            finetune_weight_decay=args.pydvl_finetune_weight_decay,
        )

        print("\n" + "-" * 72)
        print(f"Running jobs={jobs}")
        print("Command:")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        start = time.time()
        rc = subprocess.call(cmd, cwd=str(project_root))
        wall = time.time() - start

        summary_path = (
            project_root / run_output_prefix / args.head / "data_shapley" / "pydvl_data_shapley_summary.json"
        )
        summary_elapsed = 0.0
        effective_jobs = 0
        single_gpu_concurrency = False
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary_elapsed = float(summary.get("elapsed_seconds", 0.0))
            effective_jobs = int(summary.get("parallel_jobs_effective", 0))
            single_gpu_concurrency = bool(summary.get("single_gpu_concurrency_enabled", False))

        result = BenchmarkResult(
            jobs_requested=jobs,
            return_code=rc,
            wall_time_seconds=wall,
            summary_elapsed_seconds=summary_elapsed,
            parallel_jobs_effective=effective_jobs,
            single_gpu_concurrency_enabled=single_gpu_concurrency,
            summary_path=str(summary_path),
        )
        results.append(result)

        print(
            f"Result jobs={jobs}: rc={rc}, wall={wall:.1f}s, "
            f"tmc_elapsed={summary_elapsed:.1f}s, effective_jobs={effective_jobs}, "
            f"single_gpu_concurrency={single_gpu_concurrency}"
        )

    if args.dry_run:
        print("\nDry run complete.")
        return

    out_summary = output_root / "benchmark_summary.json"
    payload = {
        "dataset": args.dataset,
        "head": args.head,
        "task": args.pydvl_task,
        "jobs_list": jobs_values,
        "results": [asdict(r) for r in results],
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print("Benchmark Summary")
    print("=" * 72)
    for row in results:
        print(
            f"jobs={row.jobs_requested:<2} rc={row.return_code} "
            f"wall={row.wall_time_seconds:>8.1f}s "
            f"tmc={row.summary_elapsed_seconds:>8.1f}s "
            f"effective={row.parallel_jobs_effective:<2} "
            f"sgpu_concurrency={row.single_gpu_concurrency_enabled}"
        )
    print(f"\nSaved: {out_summary}")


if __name__ == "__main__":
    main()


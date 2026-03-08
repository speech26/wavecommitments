#!/usr/bin/env python3
"""
Run recommended pyDVL Data Shapley benchmark profiles for multi-GPU setups.

This script orchestrates multiple captum_analysis.py runs and records runtime
and effective parallelism fields from each run summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


PROFILE_SETS = {
    "safe": [
        {
            "name": "safe_8gpu_dataparallel_jobs1",
            "single_gpu": False,
            "num_gpus": 8,
            "pydvl_jobs": 1,
            "allow_single_gpu_concurrency": False,
            "pydvl_joblib_backend": "auto",
            "pydvl_finetune_epochs": 20,
            "pydvl_finetune_batch_size": 512,
            "min_updates": 50,
            "rtol": 0.08,
        },
        {
            "name": "safe_1gpu_jobs1_large_batch",
            "single_gpu": True,
            "num_gpus": 1,
            "pydvl_jobs": 1,
            "allow_single_gpu_concurrency": False,
            "pydvl_joblib_backend": "auto",
            "pydvl_finetune_epochs": 20,
            "pydvl_finetune_batch_size": 1024,
            "min_updates": 50,
            "rtol": 0.08,
        },
        {
            "name": "safe_1gpu_jobs2_overlap",
            "single_gpu": True,
            "num_gpus": 1,
            "pydvl_jobs": 2,
            "allow_single_gpu_concurrency": True,
            "pydvl_joblib_backend": "loky",
            "pydvl_finetune_epochs": 20,
            "pydvl_finetune_batch_size": 1024,
            "min_updates": 50,
            "rtol": 0.08,
        },
    ],
    "aggressive": [
        {
            "name": "aggr_1gpu_jobs4_overlap",
            "single_gpu": True,
            "num_gpus": 1,
            "pydvl_jobs": 4,
            "allow_single_gpu_concurrency": True,
            "pydvl_joblib_backend": "loky",
            "pydvl_finetune_epochs": 10,
            "pydvl_finetune_batch_size": 2048,
            "min_updates": 30,
            "rtol": 0.10,
        },
        {
            "name": "aggr_1gpu_jobs8_overlap",
            "single_gpu": True,
            "num_gpus": 1,
            "pydvl_jobs": 8,
            "allow_single_gpu_concurrency": True,
            "pydvl_joblib_backend": "loky",
            "pydvl_finetune_epochs": 8,
            "pydvl_finetune_batch_size": 2048,
            "min_updates": 20,
            "rtol": 0.12,
        },
    ],
}


@dataclass
class ProfileResult:
    profile_name: str
    return_code: int
    wall_time_seconds: float
    summary_elapsed_seconds: float
    parallel_jobs_requested: int
    parallel_jobs_effective: int
    single_gpu_concurrency_enabled: bool
    joblib_backend: str
    summary_path: str


def get_profile_runs(profile: str) -> List[Dict]:
    if profile == "all":
        return PROFILE_SETS["safe"] + PROFILE_SETS["aggressive"]
    if profile not in PROFILE_SETS:
        raise ValueError(f"Unknown profile set: {profile}")
    return PROFILE_SETS[profile]


def build_command(
    python_exe: str,
    dataset: str,
    head: str,
    pydvl_task: str,
    output_prefix: str,
    pydvl_model: str,
    pydvl_feature_head: str,
    pydvl_train_device: str,
    disable_pydvl_tqdm: bool,
    run_cfg: Dict,
) -> List[str]:
    cmd = [
        python_exe,
        "scripts/captum_analysis.py",
        "--dataset",
        dataset,
        "--head",
        head,
        "--analysis",
        "data_shapley",
        "--output",
        output_prefix,
        "--pydvl-model",
        pydvl_model,
        "--pydvl-feature-head",
        pydvl_feature_head,
        "--pydvl-task",
        pydvl_task,
        "--pydvl-train-device",
        pydvl_train_device,
        "--num-gpus",
        str(run_cfg["num_gpus"]),
        "--pydvl-jobs",
        str(run_cfg["pydvl_jobs"]),
        "--pydvl-joblib-backend",
        str(run_cfg["pydvl_joblib_backend"]),
        "--pydvl-finetune-epochs",
        str(run_cfg["pydvl_finetune_epochs"]),
        "--pydvl-finetune-batch-size",
        str(run_cfg["pydvl_finetune_batch_size"]),
        "--min-updates",
        str(run_cfg["min_updates"]),
        "--rtol",
        str(run_cfg["rtol"]),
    ]
    if run_cfg.get("single_gpu", False):
        cmd.append("--single-gpu")
    if run_cfg.get("allow_single_gpu_concurrency", False):
        cmd.append("--pydvl-allow-single-gpu-concurrency")
    if disable_pydvl_tqdm:
        cmd.append("--disable-pydvl-tqdm")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark recommended pyDVL profiles for multi-GPU environments."
    )
    parser.add_argument("--profile", choices=["safe", "aggressive", "all"], default="safe")
    parser.add_argument("--dataset", choices=["timit", "iemocap"], default="iemocap")
    parser.add_argument("--head", choices=["sid", "er", "ks", "ic"], default="er")
    parser.add_argument("--pydvl-task", choices=["auto", "speaker", "emotion"], default="emotion")
    parser.add_argument(
        "--pydvl-model",
        choices=["finetune_head", "frozen_head_top", "legacy_sklearn_mlp"],
        default="finetune_head",
    )
    parser.add_argument("--pydvl-feature-head", choices=["auto", "sid", "er", "ks", "ic"], default="auto")
    parser.add_argument("--pydvl-train-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output-root",
        default="analysis/captum_bench_profiles",
        help="Base output prefix for all profile runs",
    )
    parser.add_argument("--disable-pydvl-tqdm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs = get_profile_runs(args.profile)

    print("=" * 72)
    print("WaveCommit pyDVL Profile Benchmark")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Head: {args.head}")
    print(f"Task: {args.pydvl_task}")
    print(f"Profile set: {args.profile}")
    print(f"Run count: {len(runs)}")
    print(f"Output root: {output_root}")

    results: List[ProfileResult] = []
    for run_cfg in runs:
        profile_name = run_cfg["name"]
        output_prefix = f"{args.output_root}/{profile_name}"
        cmd = build_command(
            python_exe=sys.executable,
            dataset=args.dataset,
            head=args.head,
            pydvl_task=args.pydvl_task,
            output_prefix=output_prefix,
            pydvl_model=args.pydvl_model,
            pydvl_feature_head=args.pydvl_feature_head,
            pydvl_train_device=args.pydvl_train_device,
            disable_pydvl_tqdm=args.disable_pydvl_tqdm,
            run_cfg=run_cfg,
        )

        print("\n" + "-" * 72)
        print(f"Profile: {profile_name}")
        print("Command:")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        start = time.time()
        rc = subprocess.call(cmd, cwd=str(project_root))
        wall = time.time() - start

        summary_path = (
            project_root / output_prefix / args.head / "data_shapley" / "pydvl_data_shapley_summary.json"
        )

        summary_elapsed = 0.0
        jobs_req = 0
        jobs_eff = 0
        single_gpu_conc = False
        backend = "unknown"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary_elapsed = float(summary.get("elapsed_seconds", 0.0))
            jobs_req = int(summary.get("parallel_jobs_requested", 0))
            jobs_eff = int(summary.get("parallel_jobs_effective", 0))
            single_gpu_conc = bool(summary.get("single_gpu_concurrency_enabled", False))
            backend = str(summary.get("joblib_backend", "unknown"))

        result = ProfileResult(
            profile_name=profile_name,
            return_code=rc,
            wall_time_seconds=wall,
            summary_elapsed_seconds=summary_elapsed,
            parallel_jobs_requested=jobs_req,
            parallel_jobs_effective=jobs_eff,
            single_gpu_concurrency_enabled=single_gpu_conc,
            joblib_backend=backend,
            summary_path=str(summary_path),
        )
        results.append(result)

        print(
            f"Result rc={rc}, wall={wall:.1f}s, tmc_elapsed={summary_elapsed:.1f}s, "
            f"jobs_req={jobs_req}, jobs_eff={jobs_eff}, sgpu_overlap={single_gpu_conc}, backend={backend}"
        )

    if args.dry_run:
        print("\nDry run complete.")
        return

    out_file = output_root / f"profile_benchmark_{args.profile}.json"
    payload = {
        "dataset": args.dataset,
        "head": args.head,
        "task": args.pydvl_task,
        "profile_set": args.profile,
        "results": [asdict(r) for r in results],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 72)
    print("Benchmark Summary")
    print("=" * 72)
    for r in results:
        print(
            f"{r.profile_name:<34} rc={r.return_code} "
            f"wall={r.wall_time_seconds:>8.1f}s "
            f"tmc={r.summary_elapsed_seconds:>8.1f}s "
            f"jobs={r.parallel_jobs_requested}->{r.parallel_jobs_effective} "
            f"overlap={r.single_gpu_concurrency_enabled} backend={r.joblib_backend}"
        )
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()

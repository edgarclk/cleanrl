#!/usr/bin/env python3
"""Launch the 27-point Experiment 5 hyperparameter grid.

Grid:
    alpha      in {1e-3, 1e-2, 1e-1}
    batch_size in {50, 100, 200}
    learning_rate in {1e-4, 5e-4, 1e-3}

The same learning rate is passed to both --policy-lr and --value-lr.
Runs are resumable: an existing final model.pt for the same configuration/seed
is skipped unless --rerun is supplied.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ALPHAS = (1e-3, 1e-2, 1e-1)
BATCH_SIZES = (50, 100, 200)
LEARNING_RATES = (1e-4, 5e-4, 1e-3)


def slug_float(value: float) -> str:
    """Filesystem/experiment-name-safe float representation."""
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def parse_seeds(text: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges, e.g. 0,2,5-7."""
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return sorted(set(seeds))


@dataclass(frozen=True)
class Job:
    alpha: float
    batch_size: int
    learning_rate: float
    seed: int

    @property
    def exp_name(self) -> str:
        return (
            "fisac_grid"
            f"__a_{slug_float(self.alpha)}"
            f"__b_{self.batch_size}"
            f"__lr_{slug_float(self.learning_rate)}"
        )

    @property
    def key(self) -> str:
        return f"{self.exp_name}__seed_{self.seed}"


def existing_models(runs_dir: Path, env_id: str, job: Job) -> list[Path]:
    pattern = f"{env_id}__{job.exp_name}__{job.seed}__*/model.pt"
    return sorted(runs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)


def write_manifest_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "status",
        "alpha",
        "batch_size",
        "learning_rate",
        "seed",
        "exp_name",
        "return_code",
        "duration_seconds",
        "log_path",
        "model_path",
    ]
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_job(job: Job, args: argparse.Namespace) -> dict[str, object]:
    runs_dir = Path(args.runs_dir)
    models = existing_models(runs_dir, args.env_id, job)
    if models and not args.rerun:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "skipped_existing",
            "alpha": job.alpha,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "seed": job.seed,
            "exp_name": job.exp_name,
            "return_code": 0,
            "duration_seconds": 0.0,
            "log_path": "",
            "model_path": str(models[-1]),
        }

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.key}.log"

    command = [
        args.python,
        args.train_script,
        "--env-id",
        args.env_id,
        "--exp-name",
        job.exp_name,
        "--seed",
        str(job.seed),
        "--alpha",
        repr(job.alpha),
        "--batch-size",
        str(job.batch_size),
        "--policy-lr",
        repr(job.learning_rate),
        "--value-lr",
        repr(job.learning_rate),
        "--total-timesteps",
        str(args.total_timesteps),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--start-steps",
        str(args.start_steps),
        "--gamma",
        repr(args.gamma),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if args.no_cuda:
        command.append("--no-cuda")
    if args.capture_video:
        command.append("--capture-video")

    print(
        f"[start] alpha={job.alpha:g}, batch={job.batch_size}, "
        f"lr={job.learning_rate:g}, seed={job.seed}"
    )
    started = time.time()
    return_code = -1
    error_text = ""
    try:
        with log_path.open("w") as log_handle:
            log_handle.write("COMMAND: " + " ".join(command) + "\n\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=args.workdir,
                check=False,
                env=os.environ.copy(),
            )
            return_code = completed.returncode
    except Exception as exc:  # Keep the rest of the grid running.
        error_text = repr(exc)

    duration = time.time() - started
    models = existing_models(runs_dir, args.env_id, job)
    model_path = str(models[-1]) if models else ""
    status = "completed" if return_code == 0 and model_path else "failed"

    if error_text:
        with log_path.open("a") as log_handle:
            log_handle.write(f"\nLAUNCHER ERROR: {error_text}\n")

    print(
        f"[{status}] alpha={job.alpha:g}, batch={job.batch_size}, "
        f"lr={job.learning_rate:g}, seed={job.seed}, "
        f"duration={duration / 60:.1f} min"
    )
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "alpha": job.alpha,
        "batch_size": job.batch_size,
        "learning_rate": job.learning_rate,
        "seed": job.seed,
        "exp_name": job.exp_name,
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "log_path": str(log_path),
        "model_path": model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-script",
        default="sac_fisac_cheetah_repro_fixed.py",
        help="Path to the reproduction trainer.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--log-dir", default="grid_logs")
    parser.add_argument("--manifest", default="grid_logs/manifest.csv")
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("0"),
        help="Comma-separated seeds/ranges. Examples: 0 or 0-4 or 0,2,4.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Concurrent training processes. Use 1 on a single GPU initially.",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    args = parser.parse_args()

    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")

    workdir = Path(args.workdir).resolve()
    args.workdir = str(workdir)
    train_path = Path(args.train_script)
    if not train_path.is_absolute():
        train_path = workdir / train_path
    if not train_path.exists():
        parser.error(f"Training script not found: {train_path}")
    # Pass an absolute script path so subprocess launch is independent of cwd.
    args.train_script = str(train_path.resolve())

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = workdir / runs_dir
    args.runs_dir = str(runs_dir.resolve())

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = workdir / log_dir
    args.log_dir = str(log_dir.resolve())

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = workdir / manifest
    args.manifest = str(manifest.resolve())

    jobs = [
        Job(alpha, batch_size, learning_rate, seed)
        for seed in args.seeds
        for alpha, batch_size, learning_rate in itertools.product(
            ALPHAS, BATCH_SIZES, LEARNING_RATES
        )
    ]

    print(
        f"Launching {len(jobs)} jobs: 27 configurations x "
        f"{len(args.seeds)} seed(s)."
    )
    print(f"Parallel workers: {args.max_parallel}")
    print(f"Manifest: {args.manifest}")

    manifest_path = Path(args.manifest)
    if args.max_parallel == 1:
        for job in jobs:
            row = run_job(job, args)
            write_manifest_row(manifest_path, row)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            future_to_job: dict[Future, Job] = {
                executor.submit(run_job, job, args): job for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "status": "launcher_failed",
                        "alpha": job.alpha,
                        "batch_size": job.batch_size,
                        "learning_rate": job.learning_rate,
                        "seed": job.seed,
                        "exp_name": job.exp_name,
                        "return_code": -1,
                        "duration_seconds": 0.0,
                        "log_path": "",
                        "model_path": "",
                    }
                    print(f"[launcher_failed] {job.key}: {exc}")
                write_manifest_row(manifest_path, row)

    print("Grid launch finished. Run evaluate_fisac_grid.py to rank the models.")


if __name__ == "__main__":
    main()

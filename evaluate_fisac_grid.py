#!/usr/bin/env python3
"""Evaluate and rank checkpoints from run_fisac_grid.py.

This imports the environment and actor implementation from evaluate_fisac_v2.py,
evaluates the newest final model for each configuration/seed, and writes:
  * grid_results.csv       -- one row per trained seed
  * grid_summary.csv       -- aggregate by hyperparameter configuration
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_fisac as eval_mod


RUN_RE = re.compile(
    r"^(?P<env>.+?)__"
    r"(?P<exp>fisac_grid__a_(?P<a>[^_]+)__b_(?P<b>\d+)__lr_(?P<lr>[^_]+))__"
    r"(?P<seed>\d+)__(?P<timestamp>\d+)$"
)


def decode_slug(text: str) -> float:
    return float(text.replace("m", "-"))


def newest_final_models(runs_dir: Path) -> list[Path]:
    newest: dict[tuple[str, int], Path] = {}
    for model_path in runs_dir.glob("*__fisac_grid__*__*/model.pt"):
        match = RUN_RE.match(model_path.parent.name)
        if not match:
            continue
        key = (match.group("exp"), int(match.group("seed")))
        current = newest.get(key)
        if current is None or model_path.stat().st_mtime > current.stat().st_mtime:
            newest[key] = model_path
    return sorted(newest.values(), key=lambda p: p.parent.name)


def evaluate_model(
    model_path: Path,
    episodes: int,
    eval_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    actor_state_dict, checkpoint = eval_mod.load_actor_state_dict(str(model_path), device)
    h1, h2, checkpoint_obs_dim, checkpoint_action_dim = (
        eval_mod.infer_actor_architecture(actor_state_dict)
    )
    checkpoint_args = checkpoint.get("args", {})
    env_id = checkpoint_args.get("env_id", "HalfCheetah-v4")
    max_episode_steps = int(checkpoint_args.get("max_episode_steps", 100))

    env = eval_mod.make_env(
        env_id=env_id,
        mode="none",
        video_dir="",
        max_episode_steps=max_episode_steps,
        fisac_env=True,
    )
    env_obs_dim = int(np.prod(env.observation_space.shape))
    env_action_dim = int(np.prod(env.action_space.shape))
    if env_obs_dim != checkpoint_obs_dim or env_action_dim != checkpoint_action_dim:
        env.close()
        raise ValueError(
            f"Checkpoint/environment dimension mismatch for {model_path}: "
            f"checkpoint=({checkpoint_obs_dim},{checkpoint_action_dim}), "
            f"env=({env_obs_dim},{env_action_dim})"
        )

    env_spec = eval_mod.EnvSpec(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    actor = eval_mod.Actor(env_spec, h1, h2).to(device)
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    returns: list[float] = []
    min_margins: list[float] = []
    safe_flags: list[float] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=eval_seed + episode)
        minimum_margin = float(info.get("safety_margin", np.inf))
        episode_return = 0.0
        done = False

        while not done:
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = actor.get_mean_action(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            margin = float(info.get("safety_margin", reward))
            minimum_margin = min(minimum_margin, margin)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(episode_return)
        min_margins.append(minimum_margin)
        safe_flags.append(float(minimum_margin >= 0.0))

    env.close()

    return {
        "mean_safety_return": float(np.mean(returns)),
        "std_safety_return": float(np.std(returns)),
        "mean_min_margin": float(np.mean(min_margins)),
        "std_min_margin": float(np.std(min_margins)),
        "safe_episode_rate": float(np.mean(safe_flags)),
        "episodes": episodes,
        "eval_seed": eval_seed,
        "alpha": float(checkpoint_args.get("alpha", np.nan)),
        "batch_size": int(checkpoint_args.get("batch_size", -1)),
        "learning_rate": float(checkpoint_args.get("policy_lr", np.nan)),
        "train_seed": int(checkpoint_args.get("seed", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model_path": str(model_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--results-csv", default="grid_results.csv")
    parser.add_argument("--summary-csv", default="grid_summary.csv")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    models = newest_final_models(Path(args.runs_dir))
    if not models:
        raise SystemExit(f"No grid model.pt files found under {args.runs_dir!r}.")

    print(f"Evaluating {len(models)} model(s) on {device}...")
    result_rows: list[dict[str, Any]] = []
    for index, model_path in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] {model_path}")
        try:
            result_rows.append(
                evaluate_model(model_path, args.episodes, args.eval_seed, device)
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")

    result_rows.sort(
        key=lambda row: (
            -row["safe_episode_rate"],
            -row["mean_min_margin"],
            -row["mean_safety_return"],
        )
    )
    result_fields = [
        "alpha",
        "batch_size",
        "learning_rate",
        "train_seed",
        "global_step",
        "safe_episode_rate",
        "mean_min_margin",
        "std_min_margin",
        "mean_safety_return",
        "std_safety_return",
        "episodes",
        "eval_seed",
        "model_path",
    ]
    write_csv(Path(args.results_csv), result_rows, result_fields)

    grouped: dict[tuple[float, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["alpha"], row["batch_size"], row["learning_rate"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (alpha, batch_size, learning_rate), rows in grouped.items():
        summary_rows.append(
            {
                "alpha": alpha,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_seeds": len(rows),
                "mean_safe_episode_rate": float(
                    np.mean([row["safe_episode_rate"] for row in rows])
                ),
                "std_safe_episode_rate_across_seeds": float(
                    np.std([row["safe_episode_rate"] for row in rows])
                ),
                "mean_min_margin": float(
                    np.mean([row["mean_min_margin"] for row in rows])
                ),
                "mean_safety_return": float(
                    np.mean([row["mean_safety_return"] for row in rows])
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -row["mean_safe_episode_rate"],
            -row["mean_min_margin"],
            -row["mean_safety_return"],
        )
    )
    summary_fields = [
        "alpha",
        "batch_size",
        "learning_rate",
        "num_seeds",
        "mean_safe_episode_rate",
        "std_safe_episode_rate_across_seeds",
        "mean_min_margin",
        "mean_safety_return",
    ]
    write_csv(Path(args.summary_csv), summary_rows, summary_fields)

    print(f"\nPer-seed results: {args.results_csv}")
    print(f"Configuration summary: {args.summary_csv}")
    print("\nTop configurations:")
    for rank, row in enumerate(summary_rows[:10], start=1):
        print(
            f"{rank:2d}. alpha={row['alpha']:g}, batch={row['batch_size']}, "
            f"lr={row['learning_rate']:g}, seeds={row['num_seeds']}, "
            f"safe={100.0 * row['mean_safe_episode_rate']:.1f}%, "
            f"min_margin={row['mean_min_margin']:.4f}"
        )


if __name__ == "__main__":
    main()

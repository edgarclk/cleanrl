"""Evaluate a policy trained by sac_fisac_cheetah_repro_fixed.py.

The evaluator matches the Experiment 5 reproduction environment and actor:
  * HalfCheetah-v4 with 100-step episodes
  * standing initial condition and reset distribution
  * signed-distance reward from head, front shin, and front foot
  * [64, 32] actor architecture inferred from the checkpoint
  * old-SAC log-standard-deviation clamp to [-20, 2]

Examples:
    python evaluate_fisac.py --model-path runs/.../model.pt --render
    python evaluate_fisac.py --model-path runs/.../model.pt --record-video
"""

import argparse
import math
import os
from dataclasses import dataclass
from types import MethodType
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


@dataclass
class EnvSpec:
    """Small shim matching the vector-environment attributes used in training."""

    single_observation_space: gym.spaces.Box
    single_action_space: gym.spaces.Box


class FisacHalfCheetahWrapper(gym.Wrapper):
    """Modern Gymnasium port of the repository's cheetah_balance.py."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = self.unwrapped

        self.floor_geom_id = self._geom_id("floor")
        self.head_geom_id = self._geom_id("head")
        self.front_shin_geom_id = self._geom_id("fshin")
        self.front_foot_geom_id = self._geom_id("ffoot")

        self.torso_body_id = self._body_id("torso")
        self.front_thigh_body_id = self._body_id("fthigh")
        self.front_shin_body_id = self._body_id("fshin")
        self.front_foot_body_id = self._body_id("ffoot")

        # Standing nominal pose used by Fisac et al.
        base.init_qpos = base.init_qpos.copy()
        base.init_qpos[2] = math.radians(-70.0)
        base.init_qpos[1] += 0.05

        # Reproduce their reset distribution. In particular, root-height noise
        # is restricted to +/-0.01 so the standing model is not initialized
        # through the floor.
        def fisac_reset_model(inner_self):
            qpos = inner_self.init_qpos + inner_self.np_random.uniform(
                low=-0.1,
                high=0.1,
                size=inner_self.model.nq,
            )
            qvel = inner_self.init_qvel + 0.1 * inner_self.np_random.standard_normal(
                inner_self.model.nv
            )
            qpos[1] = inner_self.init_qpos[1] + inner_self.np_random.uniform(
                low=-0.01,
                high=0.01,
            )
            inner_self.set_state(qpos, qvel)
            return inner_self._get_obs()

        base.reset_model = MethodType(fisac_reset_model, base)

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        if geom_id < 0:
            raise ValueError(f"Could not find MuJoCo geom named {name!r}")
        return geom_id

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        if body_id < 0:
            raise ValueError(f"Could not find MuJoCo body named {name!r}")
        return body_id

    def detect_contact(self) -> bool:
        """Whether the head, front shin, or front foot contacts the floor."""
        data = self.unwrapped.data
        unsafe_geoms = {
            self.head_geom_id,
            self.front_shin_geom_id,
            self.front_foot_geom_id,
        }

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 == self.floor_geom_id and geom2 in unsafe_geoms
            ) or (
                geom2 == self.floor_geom_id and geom1 in unsafe_geoms
            ):
                return True
        return False

    def signed_distance(self) -> float:
        """Signed safety margin used in the cheetah-balance experiment."""
        if self.detect_contact():
            return -1.0

        data = self.unwrapped.data
        quat = data.xquat[self.torso_body_id]

        torso_angle = 2.0 * math.atan2(float(quat[2]), float(quat[0]))
        head_pos = (
            float(data.xpos[self.front_thigh_body_id, 2])
            + math.cos(torso_angle + 0.87) * 0.15
            - 0.046
        )
        front_shin_pos = float(data.xpos[self.front_shin_body_id, 2])
        front_foot_pos = float(data.xpos[self.front_foot_body_id, 2])

        return float(min(front_foot_pos, head_pos, front_shin_pos))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["safety_margin"] = self.signed_distance()
        return obs, info

    def step(self, action):
        obs, _mujoco_reward, terminated, truncated, info = self.env.step(action)
        margin = self.signed_distance()
        info["safety_margin"] = margin
        return obs, margin, terminated, truncated, info


class Actor(nn.Module):
    def __init__(self, env_spec: EnvSpec, hidden_size_1: int, hidden_size_2: int):
        super().__init__()

        obs_dim = int(np.prod(env_spec.single_observation_space.shape))
        action_dim = int(np.prod(env_spec.single_action_space.shape))

        self.fc1 = nn.Linear(obs_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc_mean = nn.Linear(hidden_size_2, action_dim)
        self.fc_logstd = nn.Linear(hidden_size_2, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env_spec.single_action_space.high - env_spec.single_action_space.low)
                / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env_spec.single_action_space.high + env_spec.single_action_space.low)
                / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(pre_tanh)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_mean_action(self, x):
        mean, _ = self(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias


def load_actor_state_dict(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        return checkpoint["actor"], checkpoint

    if isinstance(checkpoint, dict) and "fc1.weight" in checkpoint:
        return checkpoint, {"actor": checkpoint}

    raise ValueError(
        "Unrecognized checkpoint format. Expected either "
        "{'actor': actor.state_dict(), ...} or actor.state_dict() directly."
    )


def infer_actor_architecture(actor_state_dict: dict[str, torch.Tensor]):
    required = ("fc1.weight", "fc2.weight", "fc_mean.weight", "fc_logstd.weight")
    missing = [name for name in required if name not in actor_state_dict]
    if missing:
        raise KeyError(f"Actor state dict is missing keys: {missing}")

    hidden_size_1 = int(actor_state_dict["fc1.weight"].shape[0])
    hidden_size_2 = int(actor_state_dict["fc2.weight"].shape[0])
    obs_dim = int(actor_state_dict["fc1.weight"].shape[1])
    action_dim = int(actor_state_dict["fc_mean.weight"].shape[0])

    if int(actor_state_dict["fc2.weight"].shape[1]) != hidden_size_1:
        raise ValueError("Checkpoint has inconsistent fc1/fc2 dimensions.")
    if int(actor_state_dict["fc_mean.weight"].shape[1]) != hidden_size_2:
        raise ValueError("Checkpoint has inconsistent fc2/output dimensions.")

    return hidden_size_1, hidden_size_2, obs_dim, action_dim


def checkpoint_arg(checkpoint: dict[str, Any], name: str, default):
    checkpoint_args = checkpoint.get("args", {})
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return default


def make_env(
    env_id: str,
    mode: str,
    video_dir: str,
    max_episode_steps: int,
    fisac_env: bool,
):
    render_mode = {
        "human": "human",
        "video": "rgb_array",
        "none": None,
    }[mode]

    env = gym.make(
        env_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    if fisac_env:
        env = FisacHalfCheetahWrapper(env)

    if mode == "video":
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix="fisac-eval",
        )

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def evaluate(args):
    if args.render and args.record_video:
        raise ValueError("Use either --render or --record-video, not both.")

    mode = "human" if args.render else "video" if args.record_video else "none"
    if mode == "video":
        os.makedirs(args.video_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    actor_state_dict, checkpoint = load_actor_state_dict(args.model_path, device)
    h1, h2, checkpoint_obs_dim, checkpoint_action_dim = infer_actor_architecture(
        actor_state_dict
    )

    env_id = args.env_id or checkpoint_arg(checkpoint, "env_id", "HalfCheetah-v4")
    max_episode_steps = args.max_episode_steps
    if max_episode_steps is None:
        max_episode_steps = int(
            checkpoint_arg(checkpoint, "max_episode_steps", 100)
        )

    env = make_env(
        env_id=env_id,
        mode=mode,
        video_dir=args.video_dir,
        max_episode_steps=max_episode_steps,
        fisac_env=args.fisac_env,
    )

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("Only continuous Box action spaces are supported.")

    env_obs_dim = int(np.prod(env.observation_space.shape))
    env_action_dim = int(np.prod(env.action_space.shape))
    if env_obs_dim != checkpoint_obs_dim:
        raise ValueError(
            f"Observation dimension mismatch: checkpoint={checkpoint_obs_dim}, "
            f"environment={env_obs_dim}."
        )
    if env_action_dim != checkpoint_action_dim:
        raise ValueError(
            f"Action dimension mismatch: checkpoint={checkpoint_action_dim}, "
            f"environment={env_action_dim}."
        )

    env_spec = EnvSpec(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )
    actor = Actor(env_spec, hidden_size_1=h1, hidden_size_2=h2).to(device)
    actor.load_state_dict(actor_state_dict, strict=True)
    actor.eval()

    print(f"Loaded actor from: {args.model_path}")
    print(f"Environment: {env_id}")
    print(f"Fisac environment wrapper: {args.fisac_env}")
    print(f"Episode horizon: {max_episode_steps}")
    print(f"Actor hidden sizes: [{h1}, {h2}]")
    print(f"Device: {device}")
    print(f"Deterministic evaluation: {args.deterministic}")
    print()

    returns: list[float] = []
    lengths: list[int] = []
    minimum_margins: list[float] = []
    safe_episodes: list[float] = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        done = False
        episodic_return = 0.0
        episodic_length = 0
        minimum_margin = float(info.get("safety_margin", np.inf))

        while not done:
            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                if args.deterministic:
                    action_tensor = actor.get_mean_action(obs_tensor)
                else:
                    action_tensor, _, _ = actor.get_action(obs_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)

            done = bool(terminated or truncated)
            episodic_return += float(reward)
            episodic_length += 1

            if args.fisac_env:
                margin = float(info.get("safety_margin", reward))
                minimum_margin = min(minimum_margin, margin)

        returns.append(episodic_return)
        lengths.append(episodic_length)

        if args.fisac_env:
            minimum_margins.append(minimum_margin)
            safe = float(minimum_margin >= 0.0)
            safe_episodes.append(safe)
            print(
                f"episode={episode + 1}, "
                f"safety_return={episodic_return:.4f}, "
                f"min_margin={minimum_margin:.4f}, "
                f"safe={bool(safe)}, "
                f"length={episodic_length}"
            )
        else:
            print(
                f"episode={episode + 1}, "
                f"return={episodic_return:.2f}, "
                f"length={episodic_length}"
            )

    env.close()

    print()
    if args.fisac_env:
        print(f"Mean safety return: {np.mean(returns):.4f} +/- {np.std(returns):.4f}")
        print(
            f"Mean minimum margin: {np.mean(minimum_margins):.4f} "
            f"+/- {np.std(minimum_margins):.4f}"
        )
        print(f"Safe-episode rate: {100.0 * np.mean(safe_episodes):.1f}%")
    else:
        print(f"Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")

    print(f"Mean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")

    if args.record_video:
        print(f"Saved videos to: {args.video_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a saved model.pt checkpoint.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Environment ID. Defaults to checkpoint metadata, then HalfCheetah-v4.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Episode horizon. Defaults to checkpoint metadata, then 100.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First evaluation seed.",
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA when available.",
    )
    parser.add_argument(
        "--fisac-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the standing reset and signed-distance environment from Experiment 5.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the live MuJoCo viewer.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record every evaluation episode as video.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos/fisac_eval",
        help="Directory for recorded videos.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the squashed policy mean instead of sampled actions.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
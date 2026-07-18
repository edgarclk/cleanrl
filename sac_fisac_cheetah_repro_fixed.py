"""PyTorch/CleanRL-style reproduction of Fisac et al. Experiment 5.

This ports the relevant environment and old-SAC details from:
  HJReachability/safety_rl, branch bridging_safety_rl

It intentionally uses:
  * HalfCheetah-v4 (modern Gymnasium/MuJoCo; not bit-identical to old Gym/mujoco-py)
  * standing reset distribution from cheetah_balance.py
  * signed distance from head, front shin, and front foot
  * old SAC with twin Q networks, a separate V network, and a target V network
  * fixed gamma=0.99 and alpha=1e-3
  * [64, 32] ReLU hidden layers
  * 50,000 steps, 100-step episodes, and episode-end update bursts
"""

import math
import os
import random
import time
from dataclasses import dataclass
from types import MethodType

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    save_model: bool = True
    checkpoint_every: int = 10_000

    # Experiment 5 / final 100-seed configuration.
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 50_000
    max_episode_steps: int = 100
    num_envs: int = 1
    buffer_size: int = 1_000_000
    batch_size: int = 100
    start_steps: int = 10_000
    gamma: float = 0.99
    tau: float = 0.005  # Equivalent to old SAC polyak=0.995.
    alpha: float = 1e-3
    policy_lr: float = 1e-4
    value_lr: float = 1e-4

    # Exact experiment architecture.
    hidden_size_1: int = 64
    hidden_size_2: int = 32


class FisacHalfCheetahWrapper(gym.Wrapper):
    """Modern Gymnasium port of cheetah_balance.py."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = self.unwrapped
        model = base.model

        # Named IDs replace the old code's hard-coded and negative indices.
        self.floor_geom_id = self._geom_id("floor")
        self.head_geom_id = self._geom_id("head")
        self.front_shin_geom_id = self._geom_id("fshin")
        self.front_foot_geom_id = self._geom_id("ffoot")

        self.torso_body_id = self._body_id("torso")
        self.front_thigh_body_id = self._body_id("fthigh")
        self.front_shin_body_id = self._body_id("fshin")
        self.front_foot_body_id = self._body_id("ffoot")

        # Nominal standing initialization used in the repository.
        base.init_qpos = base.init_qpos.copy()
        base.init_qpos[2] = math.radians(-70.0)
        base.init_qpos[1] += 0.05

        # Patch the base environment's reset_model so Gymnasium's normal reset
        # path uses the same standing reset distribution as the old environment.
        def fisac_reset_model(inner_self):
            qpos = inner_self.init_qpos + inner_self.np_random.uniform(
                low=-0.1,
                high=0.1,
                size=inner_self.model.nq,
            )
            qvel = inner_self.init_qvel + 0.1 * inner_self.np_random.standard_normal(
                inner_self.model.nv
            )
            # Limit root-height noise so the standing model is not initialized
            # through the ground.
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
        """Return whether head, front shin, or front foot contacts the floor."""
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
        """Signed distance used by the original cheetah-balance experiment.

        Failure contact maps to -1. Otherwise, use the minimum of:
          * reconstructed head clearance,
          * front-shin body height,
          * front-foot body height.
        """
        if self.detect_contact():
            return -1.0

        data = self.unwrapped.data
        quat = data.xquat[self.torso_body_id]

        # The old code uses 2*atan(q_y/q_w). atan2 is numerically safer and is
        # equivalent in the orientation range used by this task.
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

        # Important: the repository computes signed_distance() after stepping,
        # so this reward is l(x_{t+1}) for transition (x_t, u_t, x_{t+1}).
        margin = self.signed_distance()
        info["safety_margin"] = margin
        return obs, margin, terminated, truncated, info


def make_env(args: Args, idx: int, run_name: str):
    def thunk():
        render_mode = "rgb_array" if args.capture_video and idx == 0 else None
        env = gym.make(
            args.env_id,
            render_mode=render_mode,
            max_episode_steps=args.max_episode_steps,
        )
        env = FisacHalfCheetahWrapper(env)
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed + idx)
        return env

    return thunk


def init_tf1_dense(layer: nn.Linear) -> None:
    """Match tf.layers.dense defaults used by the historical implementation.

    TensorFlow 1.x dense layers defaulted to Glorot/Xavier-uniform kernels and
    zero biases. PyTorch nn.Linear uses a different default initialization.
    """
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


def save_checkpoint(path, actor, qf1, qf2, vf, vf_target, args, global_step):
    torch.save(
        {
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "vf": vf.state_dict(),
            "vf_target": vf_target.state_dict(),
            "args": vars(args),
            "global_step": global_step,
        },
        path,
    )


class SoftQNetwork(nn.Module):
    def __init__(self, env, h1: int, h2: int):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            init_tf1_dense(layer)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ValueNetwork(nn.Module):
    def __init__(self, env, h1: int, h2: int):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            init_tf1_dense(layer)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class Actor(nn.Module):
    def __init__(self, env, h1: int, h2: int):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))

        self.fc1 = nn.Linear(obs_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mean = nn.Linear(h2, act_dim)
        self.fc_logstd = nn.Linear(h2, act_dim)
        for layer in (self.fc1, self.fc2, self.fc_mean, self.fc_logstd):
            init_tf1_dense(layer)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # Historical Spinning Up SAC used tanh followed by affine rescaling,
        # not direct clipping. With a near-zero raw output this initializes
        # log_std near -9 rather than near 0, a very different policy entropy.
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


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def main() -> None:
    args = tyro.cli(Args)
    if args.num_envs != 1:
        raise ValueError("This reproduction currently requires --num-envs 1.")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    actor = Actor(envs, args.hidden_size_1, args.hidden_size_2).to(device)
    qf1 = SoftQNetwork(envs, args.hidden_size_1, args.hidden_size_2).to(device)
    qf2 = SoftQNetwork(envs, args.hidden_size_1, args.hidden_size_2).to(device)
    vf = ValueNetwork(envs, args.hidden_size_1, args.hidden_size_2).to(device)
    vf_target = ValueNetwork(envs, args.hidden_size_1, args.hidden_size_2).to(device)
    vf_target.load_state_dict(vf.state_dict())
    vf_target.requires_grad_(False)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    value_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()) + list(vf.parameters()),
        lr=args.value_lr,
    )

    # Keep the environment's declared observation dtype unchanged. Mutating
    # envs.single_observation_space.dtype also mutates the space seen by
    # Gymnasium's PassiveEnvChecker, while HalfCheetah-v4 still emits float64.
    # Use a separate float32 space only for replay-buffer storage.
    replay_observation_space = gym.spaces.Box(
        low=np.asarray(envs.single_observation_space.low, dtype=np.float32),
        high=np.asarray(envs.single_observation_space.high, dtype=np.float32),
        shape=envs.single_observation_space.shape,
        dtype=np.float32,
    )
    rb = ReplayBuffer(
        args.buffer_size,
        replay_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    episode_index = 0
    episode_length = 0
    episode_return = 0.0
    episode_min_margin = float("inf")
    update_count = 0

    # Values used for periodic logging after the first update.
    last_metrics = None

    for global_step in range(args.total_timesteps):
        # Matches old SAC: uniform-random exploration for the first 10k steps,
        # while learning still takes place at episode boundaries.
        if global_step <= args.start_steps:
            actions = np.asarray([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                )
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        reward_scalar = float(rewards[0])
        episode_length += 1
        episode_return += reward_scalar
        episode_min_margin = min(episode_min_margin, reward_scalar)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Time-limit truncations are intentionally not terminal for the Bellman
        # target, matching the repository's d=False at max_ep_len behavior.
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        episode_ended = bool(terminations[0] or truncations[0])
        if episode_ended:
            episode_index += 1
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            writer.add_scalar("charts/episode_min_margin", episode_min_margin, global_step)
            writer.add_scalar(
                "charts/episode_safe",
                float(episode_min_margin >= 0.0),
                global_step,
            )
            print(
                f"step={global_step + 1}, episode={episode_index}, "
                f"return={episode_return:.4f}, min_margin={episode_min_margin:.4f}"
            )

            # The original code performs ep_len gradient steps only when an
            # episode ends. With 100-step episodes this is 100 updates/burst.
            if global_step + 1 >= args.batch_size:
                for _ in range(episode_length):
                    data = rb.sample(args.batch_size)

                    # Actor update. The old implementation uses Q1, not min(Q1,Q2),
                    # in the policy loss.
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi_for_actor = qf1(data.observations, pi)
                    actor_loss = (args.alpha * log_pi - qf1_pi_for_actor).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Q and V targets.
                    with torch.no_grad():
                        l_next = data.rewards.flatten()
                        done = data.dones.flatten()
                        v_next = vf_target(data.next_observations).flatten()
                        q_non_terminal = (
                            (1.0 - args.gamma) * l_next
                            + args.gamma * torch.minimum(l_next, v_next)
                        )
                        q_backup = done * l_next + (1.0 - done) * q_non_terminal

                        pi_for_v, log_pi_for_v, _ = actor.get_action(data.observations)
                        min_q_pi = torch.minimum(
                            qf1(data.observations, pi_for_v),
                            qf2(data.observations, pi_for_v),
                        )
                        v_backup = min_q_pi - args.alpha * log_pi_for_v

                    qf1_values = qf1(data.observations, data.actions).flatten()
                    qf2_values = qf2(data.observations, data.actions).flatten()
                    v_values = vf(data.observations)

                    # The TensorFlow code uses 0.5*mean(square); reproduce that.
                    qf1_loss = 0.5 * F.mse_loss(qf1_values, q_backup)
                    qf2_loss = 0.5 * F.mse_loss(qf2_values, q_backup)
                    v_loss = 0.5 * F.mse_loss(v_values, v_backup)
                    value_loss = qf1_loss + qf2_loss + v_loss

                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

                    polyak_update(vf, vf_target, args.tau)
                    update_count += 1

                    last_metrics = {
                        "actor_loss": actor_loss.item(),
                        "qf1_loss": qf1_loss.item(),
                        "qf2_loss": qf2_loss.item(),
                        "v_loss": v_loss.item(),
                        "qf1_value": qf1_values.mean().item(),
                        "qf2_value": qf2_values.mean().item(),
                        "v_value": v_values.mean().item(),
                        "log_pi": log_pi.mean().item(),
                        "policy_log_std": actor(data.observations)[1].mean().item(),
                    }

            episode_length = 0
            episode_return = 0.0
            episode_min_margin = float("inf")

        if global_step % 100 == 0:
            writer.add_scalar("charts/SPS", global_step / max(time.time() - start_time, 1e-9), global_step)
            writer.add_scalar("charts/updates", update_count, global_step)
            writer.add_scalar("losses/alpha", args.alpha, global_step)
            writer.add_scalar("charts/gamma", args.gamma, global_step)
            if last_metrics is not None:
                for name, value in last_metrics.items():
                    writer.add_scalar(f"losses/{name}", value, global_step)

        if (
            args.save_model
            and args.checkpoint_every > 0
            and (global_step + 1) % args.checkpoint_every == 0
        ):
            checkpoint_path = (
                f"runs/{run_name}/model_step_{global_step + 1}.pt"
            )
            save_checkpoint(
                checkpoint_path, actor, qf1, qf2, vf, vf_target, args, global_step + 1
            )
            print(f"checkpoint saved to {checkpoint_path}")

    if args.save_model:
        model_path = f"runs/{run_name}/model.pt"
        save_checkpoint(
            model_path, actor, qf1, qf2, vf, vf_target, args, args.total_timesteps
        )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
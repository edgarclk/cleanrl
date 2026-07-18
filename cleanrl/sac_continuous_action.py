# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
import math
from dataclasses import dataclass
from types import MethodType

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# Custom imports
import mujoco


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Custom options
    save_model: bool = False
    """whether to save the final model"""

    fisac_safety: bool = False
    """whether to train using Fisac-style safety margins instead of MuJoCo rewards"""

    gamma_anneal_method: str = None
    """method to anneal the discount factor"""

    gamma_end: float = 0.99999
    """target discount factor"""

    gamma_period: float = 500000
    """target discount factor"""

    max_episode_steps: int = 100
    """max timesteps per episodes"""


def configure_fisac_standing_reset(env):
    base_env = env.unwrapped

    base_env.init_qpos = base_env.init_qpos.copy()
    base_env.init_qpos[2] = np.deg2rad(-70.0)
    base_env.init_qpos[1] += 0.05

    def standing_reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1,
            high=0.1,
            size=self.model.nq,
        )

        qvel = self.init_qvel + 0.1 * self.np_random.standard_normal(
            self.model.nv
        )

        # Use much less reset noise in root height.
        qpos[1] = self.init_qpos[1] + self.np_random.uniform(
            low=-0.01,
            high=0.01,
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    base_env.reset_model = MethodType(
        standing_reset_model,
        base_env,
    )

    return env


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


def make_env(env_id,
             seed,
             idx,
             capture_video,
             run_name,
             fisac_safety,
             max_episode_steps=100):
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        env = FisacHalfCheetahWrapper(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk
    # def thunk():
    #     if capture_video and idx == 0:
    #         env = gym.make(env_id,
    #                        render_mode="rgb_array",
    #                        max_episode_steps=max_episode_steps)
    #         env = configure_fisac_standing_reset(env)
    #         if fisac_safety:
    #             env = FisacHalfCheetahWrapper(
    #                 env
    #             )
    #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    #     else:
    #         if fisac_safety:
    #             env = gym.make(env_id,
    #                            max_episode_steps=max_episode_steps)
    #             env = configure_fisac_standing_reset(env)
    #             env = FisacHalfCheetahWrapper(
    #                 env
    #             )
    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     env.action_space.seed(seed)
    #     return env

    # return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            64,
        )
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
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
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                args.fisac_safety,
                max_episode_steps=args.max_episode_steps
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                l_x = data.rewards.flatten()
                v_next = min_qf_next_target.view(-1)

                # Compute discount factor
                if args.gamma_anneal_method == "hsu2021safety_stepped":
                    c = 1. - args.gamma
                    current_epoch = global_step - args.learning_starts
                    total_epochs = args.total_timesteps - args.learning_starts
                    gamma = min(args.gamma_end, 1. - c * 2. ** (-current_epoch // int(total_epochs / 20)))
                elif args.gamma_anneal_method == "hsu2021safety_StepLRMargin":
                    numDecay = int((global_step - args.learning_starts) / args.gamma_period)
                    gamma = min(args.gamma_end, 1 - (1 - args.gamma) * (0.1 ** numDecay))
                elif args.gamma_anneal_method is None:
                    gamma = args.gamma
                else:
                    raise ValueError(f"Unknown gamma anneal method: {args.gamma_anneal_method}")

                next_q_value = (1 - gamma) * l_x + gamma * torch.minimum(l_x, v_next)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # Custom code to save final model
    if args.save_model:
        model_path = f"runs/{run_name}/model.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1.state_dict(),
                "qf2": qf2.state_dict(),
                "qf1_target": qf1_target.state_dict(),
                "qf2_target": qf2_target.state_dict(),
                "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
                "args": vars(args),
            },
            model_path,
        )
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

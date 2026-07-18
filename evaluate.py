# evaluate.py
import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class EnvSpec:
    """Small shim so this Actor matches CleanRL's vector-env-based Actor API."""
    single_observation_space: gym.spaces.Box
    single_action_space: gym.spaces.Box


class Actor(nn.Module):
    def __init__(self, env_spec: EnvSpec, hidden_size: int = 256):
        super().__init__()

        obs_dim = int(np.array(env_spec.single_observation_space.shape).prod())
        action_dim = int(np.prod(env_spec.single_action_space.shape))

        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env_spec.single_action_space.high - env_spec.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env_spec.single_action_space.high + env_spec.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


def infer_hidden_size(actor_state_dict):
    if "fc1.weight" not in actor_state_dict:
        raise KeyError("Could not find 'fc1.weight' in actor state dict.")
    return actor_state_dict["fc1.weight"].shape[0]


def load_actor_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Format from the save code we discussed:
    # torch.save({"actor": actor.state_dict(), ...}, model_path)
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        return checkpoint["actor"], checkpoint

    # Fallback: model file is directly actor.state_dict()
    if isinstance(checkpoint, dict) and "fc1.weight" in checkpoint:
        return checkpoint, {"actor": checkpoint}

    raise ValueError(
        "Unrecognized checkpoint format. Expected either "
        "{'actor': actor.state_dict(), ...} or actor.state_dict() directly."
    )


def make_env(env_id, mode, video_dir=None):
    if mode == "human":
        env = gym.make(env_id, render_mode="human")
    elif mode == "video":
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix="eval",
        )
    else:
        env = gym.make(env_id)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def evaluate(args):
    if args.render and args.record_video:
        raise ValueError("Use either --render or --record-video, not both.")

    if args.render:
        mode = "human"
    elif args.record_video:
        mode = "video"
        os.makedirs(args.video_dir, exist_ok=True)
    else:
        mode = "none"

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id, mode, args.video_dir)

    assert isinstance(env.action_space, gym.spaces.Box), "Only continuous action spaces are supported."

    actor_state_dict, checkpoint = load_actor_state_dict(args.model_path, device)

    hidden_size = args.actor_hidden_size
    if hidden_size is None:
        hidden_size = infer_hidden_size(actor_state_dict)

    env_spec = EnvSpec(
        single_observation_space=env.observation_space,
        single_action_space=env.action_space,
    )

    actor = Actor(env_spec, hidden_size=hidden_size).to(device)
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    print(f"Loaded actor from: {args.model_path}")
    print(f"Environment: {args.env_id}")
    print(f"Actor hidden size: {hidden_size}")
    print(f"Device: {device}")
    print(f"Deterministic evaluation: {args.deterministic}")
    print()

    returns = []
    lengths = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)

        done = False
        episodic_return = 0.0
        episodic_length = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                sampled_action, _, mean_action = actor.get_action(obs_tensor)

                if args.deterministic:
                    action = mean_action
                else:
                    action = sampled_action

            action = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episodic_return += float(reward)
            episodic_length += 1

            if args.render:
                env.render()

        returns.append(episodic_return)
        lengths.append(episodic_length)

        print(
            f"episode={episode + 1}, "
            f"return={episodic_return:.2f}, "
            f"length={episodic_length}"
        )

    env.close()

    print()
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

    if args.record_video:
        print(f"Saved videos to: {args.video_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model.pt checkpoint.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="HalfCheetah-v4",
        help="Gymnasium environment ID.",
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
        default=1,
        help="Evaluation seed.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show live MuJoCo viewer.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record evaluation videos.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos/eval",
        help="Directory for saved evaluation videos.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mean action instead of sampled stochastic action.",
    )
    parser.add_argument(
        "--actor-hidden-size",
        type=int,
        default=None,
        help="Actor hidden size. If omitted, inferred from checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
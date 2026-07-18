import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v4")

num_episodes = 1000

states = []
actions = []

for ep in range(num_episodes):
    print(ep)
    obs, info = env.reset(seed=ep)
    done = False

    while not done:
        # Replace this with your trained policy action later.
        action = env.action_space.sample()

        states.append(obs.copy())
        actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

states = np.asarray(states)
actions = np.asarray(actions)

print("State lower:")
print(np.min(states, axis=0))

print("State upper:")
print(np.max(states, axis=0))

def bounds(arr, inflate=1.1):
    lo = np.min(arr, axis=0)
    hi = np.max(arr, axis=0)

    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)

    lo = center - inflate * radius
    hi = center + inflate * radius

    return lo, hi

x_lo, x_hi = bounds(states)
u_lo, u_hi = bounds(actions)

print("Action lower:")
print(u_lo)

print("Action upper:")
print(u_hi)

# For actions, use the true domain, not empirical policy bounds.
u_lo = env.action_space.low
u_hi = env.action_space.high

print("State lower:")
print(x_lo)

print("State upper:")
print(x_hi)

print("Action lower:")
print(u_lo)

print("Action upper:")
print(u_hi)
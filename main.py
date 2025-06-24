import random
import gymnasium as gym
import numpy as np
import time

# Create environment with GUI (render_mode not needed during training)
env = gym.make('CartPole-v1')

# Q-learning hyperparameters
alpha = 0.9
gamma = 0.95
epsilon = 1
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 200

# Discretization settings (CartPole has 4 continuous values)
NUM_BINS = 24
obs_space_high = env.observation_space.high
obs_space_low = env.observation_space.low
obs_space_low[1] = -5  # clip velocity
obs_space_high[1] = 5
obs_space_low[3] = -5  # clip angular velocity
obs_space_high[3] = 5

bins = [
    np.linspace(obs_space_low[i], obs_space_high[i], NUM_BINS)
    for i in range(4)
]

# Discretize continuous observation
def discretize(obs):
    return tuple(
        int(np.digitize(obs[i], bins[i]) - 1)
        for i in range(4)
    )

# Create a Q-table
q_table = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n))

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    state = discretize(obs)

    for step in range(max_steps):
        # Choose action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Q-learning update
        old_value = q_table[state + (action,)]
        next_max = np.max(q_table[next_state])
        q_table[state + (action,)] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done:
            break

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

# Evaluation with rendering
env = gym.make('CartPole-v1', render_mode='human')
for episode in range(50):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False

    print(f"\n🎮 Episode: {episode}")
    for step in range(max_steps):
        env.render()
        time.sleep(0.03)  # Controls speed of animation
        action = np.argmax(q_table[state])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = discretize(next_obs)
        if done:
            print(f"✅ Finished episode {episode} with reward: {reward} in {step} steps")
            break

env.close()

import numpy as np
import random
import time

num_states = 10
actions = [0, 1]
max_steps = 50

q_table = np.zeros((num_states, len(actions)))

alpha = 0.8 # learning rate
gamma = 0.95 # future reward weight
epsilon = 1.0 # exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1
num_episodes = 500

# Training
for episode in range(num_episodes):
    state = 0
    for steps in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state])

        # Move
        next_state = max(state -1, 0) if action == 0 else min(state + 1, num_states -1)
        reward = 1 if next_state == num_states - 1 else 0

        # Update Q-table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

        if state == num_states - 1:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


# 🔍 Visualization function
def print_grid(agent_pos):
    grid = ['.'] * num_states
    grid[agent_pos] = 'A'
    grid[-1] = 'G'
    print(' '.join(grid))

# 🧪 Test the learned Q-table
state = 0
print("\n🚀 Testing Learned Policy")
print_grid(state)
time.sleep(1)

for step in range(max_steps):
    action = np.argmax(q_table[state])
    state = max(state - 1, 0) if action == 0 else min(state + 1, num_states - 1)
    print(f"\nStep {step + 1}: Move {'right' if action == 1 else 'left'}")
    print_grid(state)
    time.sleep(1)
    if state == num_states - 1:
        print("\n✅ Goal Reached!")
        break
else:
    print("\n❌ Goal Not Reached.")


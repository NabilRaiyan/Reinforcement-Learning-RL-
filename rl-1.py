import numpy as np
import random
import time

# Environment settings
num_states = 5  # positions: 0,1,2,3,4
actions = [0, 1]  # 0 = move left, 1 = move right
max_steps = 10

# Q-table: states x actions
q_table = np.zeros((num_states, len(actions)))

# Hyperparameters
alpha = 0.8  # learning rate
gamma = 0.95  # discount factor
epsilon = 0.9  # exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1

num_episodes = 1000

for episode in range(num_episodes):
    state = 0  # start at position 0

    for step in range(max_steps):
        # Choose action: explore or exploit
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state])

        # Take action, observe next state and reward
        if action == 0:
            next_state = max(state - 1, 0)  # move left
        else:
            next_state = min(state + 1, num_states - 1)  # move right

        reward = 1 if next_state == num_states - 1 else 0  # reward only at last cell

        # Update Q-table (Q-learning formula)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

        if state == num_states - 1:
            # Reached goal, end episode
            break

    # Decay exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


# Visualization helper
def print_grid(agent_pos):
    grid = ['.' for _ in range(num_states)]
    grid[agent_pos] = 'A'  # Agent
    grid[num_states - 1] = 'G'  # Goal
    print(' '.join(grid))


# After training, test the learned policy with visualization
state = 0
steps_taken = 0
print("Testing learned policy:")
print_grid(state)
time.sleep(1)

while state != num_states - 1 and steps_taken < max_steps:
    action = np.argmax(q_table[state])
    state = state + 1 if action == 1 else max(state - 1, 0)
    print(f"\nStep {steps_taken + 1}: Move {'right' if action == 1 else 'left'}")
    print_grid(state)
    steps_taken += 1
    time.sleep(1)

print("\n" + ("Goal reached!" if state == num_states - 1 else "Goal not reached."))

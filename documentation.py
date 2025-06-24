
# ------------------------- What is Reinforcement Learning (RL)? ----------------------------------------


# 1. RL is a type of AI where an agent learns to make decisions by trying actions in an environment.
# 2. The agent gets rewards or penalties based on what it does.
# 3. Goal: Learn the best actions to maximize total reward over time.

# ---------------------------
# How Does RL Work? (Simple Steps)

# 1. Agent starts in some state (situation).
# 2. Agent takes an action (a) (like move left/right).
# 3. Environment gives a reward (r) (good or bad).
# 4. Agent moves to a new state (s’).
# 5. Repeat steps — agent learns from rewards which actions are better.

# ----------------------------------------------

# What is a Q-table?
# The Q-table is like a cheat sheet or map that tells the agent:
#
# “If I am in state S, and I take action A, what is the expected reward?”
#
# It’s a table with:
# 1. States as rows
# 2. Actions as columns
#
# Each cell = Q-value (expected reward of doing action A in state S)

# ----------------------------------------------

# How Does the Q-table Get Updated?
# Initially, all Q-values are zero or random.
#
# When the agent tries an action, it updates the Q-value based on the reward it got and what it expects in the future.

# ----------------------------------------------

# Simple Q-learning Update Formula:
# Q(s,a) = Q(s,a) + α * [r + γ * max Q(s', a') - Q(s,a)]
        # Where:
        # 1. Q(s,a) = current value for state-action pair
        # 2. α = learning rate (how much to update)
        # 3. r = reward received
        # 4. γ = discount factor (importance of future rewards)
        # 5. max Q(s', a') = best future Q-value from next state

# ----------------------------------------------
# In simple words:
# 1. The agent keeps improving the Q-table by trying actions, seeing rewards, and updating its expectations.
# 2. Over time, the Q-table tells the agent the best action to take in any state.
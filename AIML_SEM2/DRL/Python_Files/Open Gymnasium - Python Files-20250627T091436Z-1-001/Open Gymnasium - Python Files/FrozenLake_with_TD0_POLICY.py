import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Environment setup ---
env = gym.make("FrozenLake-v1", is_slippery=False)
alpha = 0.1
gamma = 0.99
episodes = 1000
V = np.zeros(env.observation_space.n)

# --- TD(0) Learning ---
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state

# --- Compute Greedy Policy ---
policy = np.zeros(env.observation_space.n, dtype=int)
for state in range(env.observation_space.n):
    values = []
    for action in range(env.action_space.n):
        env.reset(seed=42)
        env.unwrapped.s = state
        next_state, reward, _, _, _ = env.step(action)
        values.append(reward + gamma * V[next_state])
    policy[state] = np.argmax(values)

print("\nGreedy policy (0=Left, 1=Down, 2=Right, 3=Up):")
print(policy.reshape(4, 4))

# --- Define FrozenLake layout ---
map_layout = [
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'H'],
    ['F', 'F', 'F', 'H'],
    ['H', 'F', 'F', 'G']
]

# --- Arrow map for directions ---
arrow_map = {
    0: '←',
    1: '↓',
    2: '→',
    3: '↑'
}

# --- Final value grid ---
value_grid = V.reshape(4, 4)

# --- Plot heatmap with policy arrows ---
plt.imshow(value_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='State Value V(s)')
plt.title("TD(0) Learned Value Function with Policy Arrows - FrozenLake")

for r in range(4):
    for c in range(4):
        state_idx = r * 4 + c
        tile = map_layout[r][c]

        if tile == 'H':
            plt.text(c, r, 'H', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        elif tile == 'G':
            plt.text(c, r, 'G', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        elif (r, c) == (0, 0):
            plt.text(c, r, '★', ha='center', va='center', color='white', fontsize=16)
        else:
            action_arrow = arrow_map.get(policy[state_idx], '')
            plt.text(c, r, action_arrow, ha='center', va='center', color='white', fontsize=16)

plt.xticks([])
plt.yticks([])
plt.show()

env.close()

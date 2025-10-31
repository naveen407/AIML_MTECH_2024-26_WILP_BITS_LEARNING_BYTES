import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Environment setup ---
# Create the FrozenLake environment with deterministic transitions (non-slippery)
env = gym.make("FrozenLake-v1", is_slippery=False)

# TD(0) hyperparameters
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor
episodes = 100    # number of training episodes

# Initialize state value table V(s)
V = np.zeros(env.observation_space.n)

# --- Create log file to save V(s) after each episode ---
log_file = open("td_learning_log.txt", "w")

# Log initial state values
log_file.write("Initial value of all states:\n")
log_file.write(np.array2string(V.reshape(4, 4), precision=2, separator=' ') + "\n\n")

# --- TD(0) learning loop ---
for ep in range(1, episodes + 1):
    state, _ = env.reset()
    done = False

    while not done:
        # Choose a random action (no policy yet)
        action = env.action_space.sample()

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # TD(0) update rule
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state

    # Log the updated value function after each episode
    log_file.write(f"Episode {ep}:\n")
    log_file.write(np.array2string(V.reshape(4, 4), precision=2, separator=' ') + "\n\n")

# --- Final logging ---
value_grid = V.reshape(4, 4)
log_file.write("Final value of all states after training:\n")
log_file.write(np.array2string(value_grid, precision=2, separator=' ') + "\n")
log_file.close()

# --- Print final state values to console ---
print("\nFinal value of all states after training:\n", np.round(value_grid, 2))

# --- FrozenLake layout (S=start, F=frozen, H=hole, G=goal) ---
map_layout = [
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'H'],
    ['F', 'F', 'F', 'H'],
    ['H', 'F', 'F', 'G']
]

# Define possible actions and their directions
actions = {
    (-1, 0): '↑',
    (1, 0): '↓',
    (0, -1): '←',
    (0, 1): '→'
}

# --- Compute greedy path from start using learned values ---
path = [(0, 0)]         # Start at (0,0)
directions = []         # To store arrows for the path
current = (0, 0)

# Try to follow the greedy path for a limited number of steps
for _ in range(20):
    r, c = current
    best_val = -1
    best_move = None

    # Look in all directions for the neighbor with highest value
    for dr, dc in actions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            if value_grid[nr][nc] > best_val:
                best_val = value_grid[nr][nc]
                best_move = (nr, nc)
                best_dir = actions[(dr, dc)]

    if best_move is None or best_move in path:
        break  # Stop if stuck or in a loop
    directions.append((best_move, best_dir))
    path.append(best_move)
    current = best_move

# --- Plot value heatmap with annotations ---
plt.imshow(value_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='State Value V(s)')
plt.title("TD(0) Learned Value Function - FrozenLake\nwith Greedy Path and Labels")

# Annotate each cell with labels (H, G, start)
for r in range(4):
    for c in range(4):
        tile = map_layout[r][c]
        if tile == 'H':
            plt.text(c, r, 'H', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        elif tile == 'G':
            plt.text(c, r, 'G', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
        elif (r, c) == (0, 0):
            plt.text(c, r, '★', ha='center', va='center', color='white', fontsize=16)

# Draw arrows showing the greedy path
for (r, c), arrow in directions:
    plt.text(c, r, arrow, ha='center', va='center', color='white', fontsize=16)

# Clean plot axes
plt.xticks([])
plt.yticks([])
plt.show()

env.close()

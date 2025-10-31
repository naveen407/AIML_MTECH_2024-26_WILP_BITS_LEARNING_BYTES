import gymnasium as gym

# Create the FrozenLake environment
# 'is_slippery=False' makes it deterministic (good for teaching TD learning)
# 'render_mode="ansi"' prints the grid to the terminal
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

# Define number of episodes (agent starts fresh each time)
num_episodes = 5

# Loop through multiple episodes
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    step = 0

    print(f"\n=== Episode {episode + 1} ===")

    # Run until episode ends (goal reached or agent falls in hole)
    while not done:
        print(env.render())  # Print the current grid state
        action = env.action_space.sample()  # Random action
        print(f"Step {step + 1}: Action taken = {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print(f"Episode ended in {step} steps. Final reward: {reward}")

env.close()

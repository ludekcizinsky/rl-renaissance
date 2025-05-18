
if __name__ == "__main__":

    import re
    import matplotlib.pyplot as plt

    # Load log file
    log_file_path = "dobro2.txt"

    # Prepare containers
    episodes = []
    min_rewards = []
    mean_rewards = []
    max_rewards = []

    # Regex to match reward line
    reward_line_re = re.compile(
        r"Episode (\d+)/\d+ - Min reward: ([\d.]+), Max reward: ([\d.]+), Mean reward: ([\d.]+)"
    )

    # Read and extract data
    with open(log_file_path, "r") as f:
        for line in f:
            match = reward_line_re.search(line)
            if match:
                episode = int(match.group(1))
                min_r = float(match.group(2))
                max_r = float(match.group(3))
                mean_r = float(match.group(4))

                episodes.append(episode)
                min_rewards.append(min_r)
                mean_rewards.append(mean_r)
                max_rewards.append(max_r)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, label='Mean Reward', color='blue')
    plt.fill_between(episodes, min_rewards, max_rewards, color='blue', alpha=0.2, label='Min/Max Reward Range')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("ppo_training_rewards2.png", dpi=300)

    

# load numpy rewards as plot
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_rewards(rewards_file, save_dir):
    # Load the rewards from the file
    rewards = np.load(rewards_file)
    last_reward = rewards[:,-1]
    print("rewards shape:", rewards.shape)

    # Create a directory to save the plots if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(last_reward, label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'rewards_plot0.png')
    plt.savefig(plot_path)
    print(f"Plot saved at {plot_path}")

if __name__ == "__main__":

    
    rewards_file = "/home/renaissance/work/output/run_0/rewards.npy"
    save_dir = "/home/renaissance/work/output/plot_rewards"

    plot_rewards(rewards_file, save_dir)



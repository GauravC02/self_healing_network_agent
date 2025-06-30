import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from network.network_environment import NetworkEnvironment

class NetworkVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.performance_data = []
        self.reward_data = []
        self.steps = []
        self.step_count = 0
        
        # Set style
        plt.style.use('seaborn')
        self.fig.suptitle('Network Performance Monitoring', fontsize=14)
        
        # Configure plots
        self.ax1.set_title('Network Performance Over Time')
        self.ax1.set_ylabel('Performance (%)')
        self.ax1.grid(True)
        
        self.ax2.set_title('Reward History')
        self.ax2.set_ylabel('Reward')
        self.ax2.grid(True)
        
        plt.tight_layout()
        
    def update(self, performance, reward):
        try:
            # Convert inputs to float
            performance = float(performance)
            reward = float(reward)
            
            self.step_count += 1
            self.steps.append(self.step_count)
            self.performance_data.append(performance)
            self.reward_data.append(reward)

            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()

            # Plot performance
            self.ax1.plot(self.steps, self.performance_data, 'b-', label='Performance', linewidth=2)
            self.ax1.set_title('Network Performance Over Time')
            self.ax1.set_xlabel('Steps')
            self.ax1.set_ylabel('Performance (%)')
            self.ax1.set_ylim(0, 100)  # Performance is percentage
            self.ax1.grid(True)
            self.ax1.legend()

            # Plot reward
            self.ax2.plot(self.steps, self.reward_data, 'g-', label='Reward', linewidth=2)
            self.ax2.set_title('Reward Over Time')
            self.ax2.set_xlabel('Steps')
            self.ax2.set_ylabel('Reward')
            self.ax2.grid(True)
            self.ax2.legend()

            # Adjust y-axis limits for reward dynamically
            if len(self.reward_data) > 0:
                reward_min = min(self.reward_data)
                reward_max = max(self.reward_data)
                margin = (reward_max - reward_min) * 0.1
                self.ax2.set_ylim(reward_min - margin, reward_max + margin)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            # Print current values
            print(f"\rCurrent Performance: {performance:.2f}% | Current Reward: {reward:.2f}", end="")
            
        except Exception as e:
            print(f"\nError updating visualization: {str(e)}")
            pass

def collect_dataset(episodes=5, steps_per_episode=50, output_file='network_dqn_dataset.csv', visualize=True):
    env = NetworkEnvironment()
    visualizer = NetworkVisualizer() if visualize else None
    print("Initialized network environment and visualizer")
    
    # Create the CSV file and write headers
    os.makedirs("dataset", exist_ok=True)
    filepath = os.path.join("dataset", output_file)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = [f'state_{i}' for i in range(env.state_space)]
        header += ['action', 'reward', 'performance', 'performance_change', 'baseline_improvement']
        header += [f'next_state_{i}' for i in range(env.state_space)]
        header += ['done']
        writer.writerow(header)
        
        print("\nStarting data collection...")
        print(f"Episodes: {episodes}, Steps per episode: {steps_per_episode}")
        print("Initial baseline performance will be calculated...\n")

        for ep in range(episodes):
            print(f"\n=== Episode {ep+1}/{episodes} ===")
            state = env.reset()
            episode_rewards = []
            episode_performance = []
            episode_improvements = []

            for step in range(steps_per_episode):
                # Choose random action for data collection
                action = np.random.randint(env.action_space)
                
                # Take a step
                next_state, reward, done, info = env.step(action)
                
                # Extract all metrics from info
                performance = info.get('performance', 0.0)
                performance_change = info.get('performance_change', 0.0)
                baseline_improvement = info.get('baseline_improvement', 0.0)

                # Update visualization with current metrics
                if visualizer:
                    visualizer.update(performance, reward)

                # Log row with all metrics
                row = state.tolist() + [
                    action,
                    reward,
                    performance,
                    performance_change,
                    baseline_improvement
                ] + next_state.tolist() + [int(done)]
                writer.writerow(row)

                # Track episode metrics
                episode_rewards.append(reward)
                episode_performance.append(performance)
                episode_improvements.append(baseline_improvement)

                # Print detailed step information
                print(f"\nStep {step+1}/{steps_per_episode}:")
                print(f"  Action: {action}")
                print(f"  Performance: {performance:.2f}% (Change: {performance_change:+.2f}%)")
                print(f"  Baseline Improvement: {baseline_improvement:+.2f}%")
                print(f"  Reward: {reward:.2f}")

                state = next_state
                if done:
                    break

            # Print episode summary
            avg_performance = np.mean(episode_performance)
            avg_reward = np.mean(episode_rewards)
            avg_improvement = np.mean(episode_improvements)
            print(f"\nEpisode {ep+1} Summary:")
            print(f"  Average Performance: {avg_performance:.2f}%")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Improvement: {avg_improvement:+.2f}%")

    env.cleanup()
    print(f"\n✅ Dataset collection complete. Saved to: {filepath}")
    
    if visualizer:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    collect_dataset(episodes=10, steps_per_episode=50)

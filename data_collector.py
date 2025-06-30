import csv
import numpy as np
import os
from env.mininet_env import MininetNetworkEnvironment

def collect_dataset(episodes=5, steps_per_episode=50, output_file='network_dqn_dataset.csv'):
    env = MininetNetworkEnvironment()
    
    # Create the CSV file and write headers
    os.makedirs("dataset", exist_ok=True)
    filepath = os.path.join("dataset", output_file)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = [f'state_{i}' for i in range(env.state_space)]
        header += ['action', 'reward']
        header += [f'next_state_{i}' for i in range(env.state_space)]
        header += ['done']
        writer.writerow(header)

        for ep in range(episodes):
            print(f"\n--- Episode {ep+1} ---")
            state = env.reset()

            for step in range(steps_per_episode):
                # Choose random action for data collection
                action = np.random.randint(env.action_space)
                
                # Take a step
                next_state, reward, done, info = env.step(action)

                # Log row as: state + action + reward + next_state + done
                row = state.tolist() + [action, reward] + next_state.tolist() + [int(done)]
                writer.writerow(row)

                print(f"Step {step+1}/{steps_per_episode}: Action={action}, Reward={reward}, Done={done}")

                state = next_state

                if done:
                    break

    env.cleanup()
    print(f"\n✅ Dataset collection complete. Saved to: {filepath}")

if __name__ == "__main__":
    collect_dataset(episodes=10, steps_per_episode=50)

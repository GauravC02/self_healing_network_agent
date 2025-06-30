import os
import numpy as np
from network.network_environment import NetworkEnvironment
from agent.dqn_agent import DQNAgent
import torch
import json
from datetime import datetime
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def train_agent(episodes=1000, max_steps=50, save_interval=10):
    setup_logging()
    logging.info("Starting training process")
    
    # Create environment and agent
    env = NetworkEnvironment()
    agent = DQNAgent(state_size=env.state_space, action_size=env.action_space)
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'network_performance': [],
        'healing_actions': [],
        'training_losses': [],
        'action_success_rate': [0] * env.action_space,
        'action_counts': [0] * env.action_space
    }
    
    best_avg_reward = float('-inf')
    best_performance = float('-inf')
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        for episode in range(episodes):
            logging.info(f"\nStarting episode {episode + 1}/{episodes}")
            
            state = env.reset()
            episode_reward = 0
            actions_taken = []
            episode_losses = []
            episode_performances = []
            
            for step in range(max_steps):
                try:
                    # Choose and perform action
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    
                    # Update action statistics
                    metrics['action_counts'][action] += 1
                    if reward > 0:
                        metrics['action_success_rate'][action] += 1
                    
                    # Store experience and train
                    agent.remember(state, action, reward, next_state, done)
                    loss = agent.replay()
                    
                    if loss is not None and loss > 0:
                        episode_losses.append(loss)
                    
                    # Track metrics
                    episode_reward += reward
                    actions_taken.append(action)
                    episode_performances.append(info['performance'])
                    state = next_state
                    
                    # Log step information with detailed metrics
                    logging.info(
                        f"Step {step + 1}: Action={action}, Reward={reward:.2f}, "
                        f"Performance={info['performance']:.2f}%, "
                        f"Action Reward={info.get('action_reward', 0):.2f}, "
                        f"State Improvement={info.get('state_improvement', 0):.2f}"
                    )
                    
                    if done:
                        break
                        
                except Exception as e:
                    logging.error(f"Error during step execution: {str(e)}")
                    continue
            
            # Calculate episode statistics
            episode_avg_performance = np.mean(episode_performances) if episode_performances else 0
            episode_max_performance = np.max(episode_performances) if episode_performances else 0
            
            # Update metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['network_performance'].append(episode_avg_performance)
            metrics['healing_actions'].append(actions_taken)
            metrics['training_losses'].extend(episode_losses)
            
            # Calculate success rates
            for i in range(env.action_space):
                if metrics['action_counts'][i] > 0:
                    success_rate = metrics['action_success_rate'][i] / metrics['action_counts'][i]
                    logging.info(f"Action {i} success rate: {success_rate:.2%}")
            
            # Calculate averages
            avg_reward = np.mean(metrics['episode_rewards'][-100:]) if metrics['episode_rewards'] else 0
            avg_performance = np.mean(metrics['network_performance'][-100:]) if metrics['network_performance'] else 0
            
            # Save best models based on both reward and performance
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(model_dir, 'best_reward_model.pth'))
                logging.info(f"New best reward model saved with average reward: {avg_reward:.2f}")
            
            if episode_max_performance > best_performance:
                best_performance = episode_max_performance
                agent.save(os.path.join(model_dir, 'best_performance_model.pth'))
                logging.info(f"New best performance model saved with performance: {best_performance:.2f}%")
            
            # Save checkpoint periodically
            if episode % save_interval == 0:
                checkpoint_path = os.path.join(model_dir, f'checkpoint_episode_{episode}.pth')
                agent.save(checkpoint_path)
                
                # Save detailed metrics
                metrics_path = os.path.join(model_dir, f'metrics_episode_{episode}.json')
                detailed_metrics = {
                    **metrics,
                    'best_reward': best_avg_reward,
                    'best_performance': best_performance,
                    'current_avg_reward': avg_reward,
                    'current_avg_performance': avg_performance,
                    'action_success_rates': [
                        metrics['action_success_rate'][i] / max(metrics['action_counts'][i], 1)
                        for i in range(env.action_space)
                    ]
                }
                
                with open(metrics_path, 'w') as f:
                    json.dump(detailed_metrics, f, indent=4)
                
                logging.info(f"Checkpoint saved at episode {episode}")
                logging.info(f"Current average reward: {avg_reward:.2f}")
                logging.info(f"Current average performance: {avg_performance:.2f}%")
            
            # Log episode summary
            logging.info(f"Episode {episode + 1} Summary:")
            logging.info(f"Total Reward: {episode_reward:.2f}")
            if episode_losses:
                logging.info(f"Average Loss: {np.mean(episode_losses):.4f}")
            logging.info(f"Final Performance: {episode_max_performance:.2f}%")
            logging.info(f"Epsilon: {agent.epsilon:.4f}")
    
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user")
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
    
    finally:
        # Save final model and metrics
        try:
            final_model_path = os.path.join(model_dir, 'final_model.pth')
            agent.save(final_model_path)
            
            final_metrics_path = os.path.join(model_dir, 'final_metrics.json')
            with open(final_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logging.info("\nTraining completed!")
            logging.info(f"Final model saved to: {final_model_path}")
            logging.info(f"Final metrics saved to: {final_metrics_path}")
        
        except Exception as e:
            logging.error(f"Error saving final results: {str(e)}")
        
        # Cleanup
        env.cleanup()

if __name__ == "__main__":
    train_agent(episodes=1000, max_steps=50, save_interval=10)
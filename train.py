import os
import numpy as np
from network.network_environment import NetworkEnvironment
from agent.dqn_agent import DQNAgent
import torch
import json
from datetime import datetime
import logging

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set debug level for specific loggers
    logging.getLogger('network.network_monitor').setLevel(logging.DEBUG)
    logging.getLogger('network.network_environment').setLevel(logging.DEBUG)
    
    return logging.getLogger(__name__)

def train_agent(episodes=1000, max_steps=50, save_interval=10):
    logger = setup_logging()
    logger.info("Starting training process with enhanced monitoring")
    
    # Log training parameters
    logger.info(f"Training Parameters:")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Max Steps per Episode: {max_steps}")
    logger.info(f"Save Interval: {save_interval}")
    logger.info(f"Using PyTorch Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    
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
                    
                    # Log step information with enhanced metrics
                    logger.info(
                        f"Step {step + 1}: Action={action}, Reward={reward:.2f}, "
                        f"Performance={info['performance']:.2f}%, "
                        f"Best Performance={info['best_performance']:.2f}%, "
                        f"Performance Change={info['performance_change']*100:.2f}%, "
                        f"Baseline Improvement={info['baseline_improvement']*100:.2f}%, "
                        f"Relative to Best={info['relative_to_best']*100:.2f}%, "
                        f"State Improvement={info['state_improvement']:.2f}"
                    )
                    
                    if done:
                        break
                        
                except Exception as e:
                    logger.error(f"Error during step execution: {str(e)}")
                    continue
            
            # Calculate enhanced episode statistics
            episode_avg_performance = np.mean(episode_performances) if episode_performances else 0
            episode_max_performance = np.max(episode_performances) if episode_performances else 0
            episode_min_performance = np.min(episode_performances) if episode_performances else 0
            performance_improvement = episode_max_performance - episode_min_performance
            
            # Update metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['network_performance'].append(episode_avg_performance)
            metrics['healing_actions'].append(actions_taken)
            metrics['training_losses'].extend(episode_losses)
            
            # Calculate success rates
            for i in range(env.action_space):
                if metrics['action_counts'][i] > 0:
                    success_rate = metrics['action_success_rate'][i] / metrics['action_counts'][i]
                    logger.info(f"Action {i} success rate: {success_rate:.2%}")
            
            # Calculate averages
            avg_reward = np.mean(metrics['episode_rewards'][-100:]) if metrics['episode_rewards'] else 0
            avg_performance = np.mean(metrics['network_performance'][-100:]) if metrics['network_performance'] else 0
            
            # Save best models based on both reward and performance
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(model_dir, 'best_reward_model.pth'))
                logger.info(f"New best reward model saved with average reward: {avg_reward:.2f}")
            
            if episode_max_performance > best_performance:
                best_performance = episode_max_performance
                agent.save(os.path.join(model_dir, 'best_performance_model.pth'))
                logger.info(f"New best performance model saved with performance: {best_performance:.2f}%")
            
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
                
                logger.info(f"Checkpoint saved at episode {episode}")
                logger.info(f"Current average reward: {avg_reward:.2f}")
                logger.info(f"Current average performance: {avg_performance:.2f}%")
            
            # Log enhanced episode summary
            logger.info(f"Episode {episode + 1} Summary:")
            logger.info(f"Total Reward: {episode_reward:.2f}")
            logger.info(f"Performance Metrics:")
            logger.info(f"  - Maximum: {episode_max_performance:.2f}%")
            logger.info(f"  - Average: {episode_avg_performance:.2f}%")
            logger.info(f"  - Minimum: {episode_min_performance:.2f}%")
            logger.info(f"  - Improvement: {performance_improvement:.2f}%")
            if episode_losses:
                logger.info(f"Training Metrics:")
                logger.info(f"  - Average Loss: {np.mean(episode_losses):.4f}")
                logger.info(f"  - Min Loss: {np.min(episode_losses):.4f}")
                logger.info(f"  - Max Loss: {np.max(episode_losses):.4f}")
            logger.info(f"Agent State:")
            logger.info(f"  - Epsilon: {agent.epsilon:.4f}")
            logger.info(f"  - Memory Size: {len(agent.memory)}")
            logger.info(f"  - Learning Rate: {agent.learning_rate:.6f}")
    
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
    
    finally:
        # Save final model and metrics
        try:
            final_model_path = os.path.join(model_dir, 'final_model.pth')
            agent.save(final_model_path)
            
            final_metrics_path = os.path.join(model_dir, 'final_metrics.json')
            with open(final_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info("\nTraining completed!")
            logger.info(f"Final model saved to: {final_model_path}")
            logger.info(f"Final metrics saved to: {final_metrics_path}")
        
        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
        
        # Cleanup
        env.cleanup()

if __name__ == "__main__":
    train_agent(episodes=1000, max_steps=50, save_interval=10)
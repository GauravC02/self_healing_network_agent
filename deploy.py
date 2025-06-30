import os
import time
import torch
import logging
from network.network_environment import NetworkEnvironment
from agent.dqn_agent import DQNAgent
from datetime import datetime
import json
import signal
import sys
from visualizer import NetworkVisualizer

class NetworkHealer:
    def __init__(self, model_path='models/best_model.pth', visualize=True):
        # Setup logging
        self.setup_logging()
        
        # Initialize environment and agent
        self.env = NetworkEnvironment()
        self.agent = DQNAgent(state_size=self.env.state_space, action_size=self.env.action_space)
        
        # Load trained model
        if os.path.exists(model_path):
            self.agent.load(model_path)
            logging.info(f"Loaded trained model from {model_path}")
        else:
            raise FileNotFoundError(f"No trained model found at {model_path}")
        
        # Performance tracking
        self.performance_history = []
        self.action_history = []
        self.healing_events = []
        
        # Initialize visualizer
        self.visualizer = NetworkVisualizer() if visualize else None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('network_healing.log'),
                logging.StreamHandler()
            ]
        )
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logging.info("\nReceived interrupt signal. Cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def monitor_and_heal(self, monitoring_interval=5, performance_threshold=0.8):
        """Continuously monitor network and apply healing actions when needed"""
        try:
            # Initialize network monitoring
            state = self.env.reset()
            baseline_performance = self.env.baseline_performance
            logging.info(f"Network monitoring initialized. Baseline performance: {baseline_performance:.2f}%")
            
            while True:
                # Get current network state and performance
                current_performance = self.env.monitor.calculate_performance_score()
                self.performance_history.append(current_performance)
                
                # Update visualization
                if self.visualizer:
                    self.visualizer.update(current_performance, 0)  # 0 as default reward when no action taken
                
                # Log current network status
                self.log_network_status(current_performance)
                
                # Check if healing is needed
                if current_performance < baseline_performance * performance_threshold:
                    self.handle_performance_degradation(state, current_performance)
                    
                    # Get new state after healing action
                    state = self.env.monitor.get_network_state()
                else:
                    logging.info(f"Network performance stable at {current_performance:.2f}%")
                
                # Save monitoring data periodically
                if len(self.performance_history) % 60 == 0:  # Save every 60 cycles
                    self.save_monitoring_data()
                
                # Wait before next monitoring cycle
                time.sleep(monitoring_interval)
        
        except Exception as e:
            logging.error(f"Error during monitoring: {str(e)}")
        finally:
            self.cleanup()
    
    def handle_performance_degradation(self, state, current_performance):
        """Handle detected performance degradation"""
        logging.warning(f"Performance degradation detected: {current_performance:.2f}%")
        
        # Get healing action from agent
        action = self.agent.choose_action(state, training=False)
        
        # Apply healing action
        new_state, reward, _, info = self.env.step(action)
        
        # Record healing event
        healing_event = {
            'timestamp': datetime.now().isoformat(),
            'performance_before': current_performance,
            'performance_after': info['performance'],
            'action_taken': action,
            'action_name': list(self.env.actions.keys())[action],
            'reward': reward
        }
        self.healing_events.append(healing_event)
        
        # Log healing action and results
        logging.info(f"Applied healing action: {list(self.env.actions.keys())[action]}")
        logging.info(f"Performance improved from {current_performance:.2f}% to {info['performance']:.2f}%")
        
        # Update visualization with new performance and reward
        if self.visualizer:
            self.visualizer.update(info['performance'], reward)
    
    def log_network_status(self, current_performance):
        """Log detailed network status"""
        network_stats = self.env.monitor.network_stats
        
        logging.info("\nNetwork Status Report:")
        logging.info(f"Overall Performance: {current_performance:.2f}%")
        
        for iface, stats in network_stats.items():
            logging.info(f"\nInterface: {iface}")
            for metric, value in stats.items():
                logging.info(f"{metric}: {value:.2f}")
    
    def save_monitoring_data(self):
        """Save monitoring and healing data to file"""
        monitoring_data = {
            'performance_history': self.performance_history,
            'healing_events': self.healing_events,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'monitoring_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(monitoring_data, f, indent=4)
        
        logging.info(f"Monitoring data saved to {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.save_monitoring_data()
        self.env.cleanup()
        logging.info("Cleanup completed")

def main():
    try:
        # Path to your trained model
        model_path = 'models/best_model.pth'
        
        # Create and start network healer with visualization
        healer = NetworkHealer(model_path, visualize=True)
        healer.monitor_and_heal()
    
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import time
import logging
import subprocess
import numpy as np
from network.network_monitor import NetworkMonitor

class NetworkEnvironment:
    def __init__(self):
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.monitor = NetworkMonitor()
        self.action_space = 8
        self.state_space = 80
        self.episode_length = 50
        self.current_step = 0
        self.baseline_performance = 1.0  # Set minimum baseline
        self.best_performance = 1.0      # Track best performance
        
        # Action mapping
        self.actions = {
            0: self.optimize_routing,
            1: self.manage_qos,
            2: self.reset_interface,
            3: self.clear_cache,
            4: self.optimize_tcp,
            5: self.balance_load,
            6: self.flush_dns,
            7: self.do_nothing
        }
        
        # Initialize network monitoring
        self.monitor.initialize()
        self.monitor.start_monitoring()
        time.sleep(2)  # Allow initial metrics collection
        initial_score = self.monitor.calculate_performance_score()
        self.baseline_performance = max(1.0, initial_score)
        self.best_performance = self.baseline_performance
    
    def reset(self):
        """Reset the environment to initial state while preserving performance history"""
        self.current_step = 0
        
        # Store best performance before reset
        previous_best = self.best_performance
        
        # Reset monitoring
        self.monitor.stop_monitoring()
        self.monitor.initialize()
        self.monitor.start_monitoring()
        time.sleep(2)  # Allow metrics to stabilize
        
        # Get initial performance after reset
        initial_performance = max(1.0, self.monitor.calculate_performance_score())
        
        # Update baseline and best performance
        self.baseline_performance = initial_performance
        self.best_performance = max(previous_best, initial_performance)
        
        self.logger.info(f"Environment reset - Initial Performance: {initial_performance:.2f}%, Best Performance: {self.best_performance:.2f}%")
        
        return self.monitor.get_network_state()
    
    def step(self, action):
        """Execute one environment step with improved reward calculation"""
        self.current_step += 1
        
        # Get initial state and performance
        initial_state = self.monitor.get_network_state()
        initial_performance = max(1.0, self.monitor.calculate_performance_score())
        
        # Execute the selected action
        action_result = self.actions[action]()
        time.sleep(2)  # Allow more time for action to take effect
        
        # Get new state and performance
        new_state = self.monitor.get_network_state()
        current_performance = max(1.0, self.monitor.calculate_performance_score())
        
        # Update best performance
        if current_performance > self.best_performance:
            self.best_performance = current_performance
        
        # Calculate various reward components
        performance_change = (current_performance - initial_performance) / initial_performance
        baseline_improvement = (current_performance - self.baseline_performance) / self.baseline_performance
        relative_to_best = (current_performance - self.best_performance) / self.best_performance
        state_improvement = np.mean(new_state) - np.mean(initial_state)
        
        # Calculate final reward with relative improvements
        reward = 0
        reward += performance_change * 3      # Immediate relative improvement
        reward += baseline_improvement * 2    # Long-term relative improvement
        reward += relative_to_best           # Reward for achieving new best
        reward += state_improvement          # State improvement
        
        # Add action-specific rewards
        if action == 7:  # do_nothing
            if current_performance >= self.best_performance * 0.95:  # Within 5% of best
                reward += 0.5  # Small reward for maintaining good performance
            else:
                reward -= 0.5  # Small penalty for doing nothing when improvement needed
        
        # Add performance threshold rewards
        if current_performance > initial_performance * 1.1:  # 10% improvement
            reward += 1
        if current_performance > self.best_performance:
            reward += 2  # Bonus for new best performance
        elif current_performance < initial_performance * 0.9:  # 10% degradation
            reward -= 1
        
        # Normalize reward
        reward = np.clip(reward, -5, 5)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Create detailed info dictionary
        info = {
            'performance': current_performance,
            'best_performance': self.best_performance,
            'performance_change': performance_change,
            'baseline_improvement': baseline_improvement,
            'relative_to_best': relative_to_best,
            'state_improvement': state_improvement,
            'action_reward': reward
        }
        
        self.logger.info(f"Step {self.current_step}: Performance={current_performance:.2f}%, Best={self.best_performance:.2f}%, Reward={reward:.2f}")
        return new_state, reward, done, info
    
    def cleanup(self):
        """Cleanup environment resources"""
        self.monitor.stop_monitoring()
    
    def optimize_routing(self):
        """Optimize network routing"""
        try:
            # Flush routing table and reset network adapter
            subprocess.run(['ipconfig', '/flushdns'], shell=True, check=True)
            subprocess.run(['route', 'FLUSH'], shell=True, check=True)
            self.logger.info("Routing optimization completed")
        except Exception as e:
            self.logger.error(f"Error optimizing routing: {str(e)}")
    
    def manage_qos(self):
        """Manage Quality of Service settings"""
        try:
            # Reset QoS policies using PowerShell
            ps_command = 'Get-NetQosPolicy | Remove-NetQosPolicy -Confirm:$false; New-NetQosPolicy -Name "DefaultPolicy" -Default'
            subprocess.run(['powershell', '-Command', ps_command], shell=True, check=True)
            self.logger.info("QoS management completed")
        except Exception as e:
            self.logger.error(f"Error managing QoS: {str(e)}")
    
    def reset_interface(self):
        """Reset network interfaces"""
        try:
            # Disable and enable network adapters
            subprocess.run(['netsh', 'interface', 'set', 'interface', 'name=*', 'admin=disabled'], shell=True, check=True)
            time.sleep(2)
            subprocess.run(['netsh', 'interface', 'set', 'interface', 'name=*', 'admin=enabled'], shell=True, check=True)
            self.logger.info("Network interface reset completed")
        except Exception as e:
            self.logger.error(f"Error resetting interface: {str(e)}")
    
    def clear_cache(self):
        """Clear network-related caches"""
        try:
            # Clear DNS and ARP caches
            subprocess.run(['ipconfig', '/flushdns'], shell=True, check=True)
            subprocess.run(['arp', '-d', '*'], shell=True, check=True)
            self.logger.info("Cache clearing completed")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
    
    def optimize_tcp(self):
        """Optimize TCP settings"""
        try:
            # Reset TCP/IP stack
            subprocess.run(['netsh', 'int', 'ip', 'reset'], shell=True, check=True)
            subprocess.run(['netsh', 'winsock', 'reset'], shell=True, check=True)
            self.logger.info("TCP optimization completed")
        except Exception as e:
            self.logger.error(f"Error optimizing TCP: {str(e)}")
    
    def balance_load(self):
        """Balance network load"""
        try:
            # Reset network adapter binding order
            ps_command = 'Get-NetAdapter | Sort-Object -Property Speed -Descending | Set-NetIPInterface -InterfaceMetric 0'
            subprocess.run(['powershell', '-Command', ps_command], shell=True, check=True)
            self.logger.info("Load balancing completed")
        except Exception as e:
            self.logger.error(f"Error balancing load: {str(e)}")
    
    def flush_dns(self):
        """Flush DNS cache"""
        try:
            subprocess.run(['ipconfig', '/flushdns'], shell=True, check=True)
            self.logger.info("DNS flush completed")
        except Exception as e:
            self.logger.error(f"Error flushing DNS: {str(e)}")
    
    def do_nothing(self):
        """No action"""
        pass
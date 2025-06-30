import time
import logging
import subprocess
import numpy as np
from network.network_monitor import NetworkMonitor

class NetworkEnvironment:
    def __init__(self):
        self.monitor = NetworkMonitor()
        self.action_space = 8
        self.state_space = 80
        self.episode_length = 50
        self.current_step = 0
        self.baseline_performance = 0
        
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
        self.baseline_performance = self.monitor.calculate_performance_score()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.monitor.stop_monitoring()
        self.monitor.initialize()
        self.monitor.start_monitoring()
        time.sleep(1)  # Allow metrics to stabilize
        return self.monitor.get_network_state()
    
    def step(self, action):
        """Execute one environment step with improved reward calculation"""
        self.current_step += 1
        
        # Get initial state and performance
        initial_state = self.monitor.get_network_state()
        initial_performance = self.monitor.calculate_performance_score()
        
        # Execute the selected action
        action_result = self.actions[action]()
        time.sleep(2)  # Allow more time for action to take effect
        
        # Get new state and performance
        new_state = self.monitor.get_network_state()
        current_performance = self.monitor.calculate_performance_score()
        
        # Calculate various reward components
        performance_change = current_performance - initial_performance
        baseline_improvement = current_performance - self.baseline_performance
        state_improvement = np.mean(new_state) - np.mean(initial_state)
        
        # Calculate final reward
        reward = 0
        reward += performance_change * 2  # Immediate performance improvement
        reward += baseline_improvement    # Long-term improvement from baseline
        reward += state_improvement       # State improvement
        
        # Add bonuses/penalties based on performance thresholds
        if current_performance > 80:
            reward += 2  # Bonus for high performance
        elif current_performance < 20:
            reward -= 2  # Penalty for very low performance
        
        # Normalize reward
        reward = np.clip(reward, -10, 10)
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Create detailed info dictionary
        info = {
            'performance': current_performance,
            'performance_change': performance_change,
            'baseline_improvement': baseline_improvement,
            'state_improvement': state_improvement,
            'action_reward': reward
        }
        
        logging.info(f"Step {self.current_step}: Performance={current_performance:.2f}%, Reward={reward:.2f}")
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
            logging.info("Routing optimization completed")
        except Exception as e:
            logging.error(f"Error optimizing routing: {str(e)}")
    
    def manage_qos(self):
        """Manage Quality of Service settings"""
        try:
            # Reset QoS policies using PowerShell
            ps_command = 'Get-NetQosPolicy | Remove-NetQosPolicy -Confirm:$false; New-NetQosPolicy -Name "DefaultPolicy" -Default'
            subprocess.run(['powershell', '-Command', ps_command], shell=True, check=True)
            logging.info("QoS management completed")
        except Exception as e:
            logging.error(f"Error managing QoS: {str(e)}")
    
    def reset_interface(self):
        """Reset network interfaces"""
        try:
            # Disable and enable network adapters
            subprocess.run(['netsh', 'interface', 'set', 'interface', 'name=*', 'admin=disabled'], shell=True, check=True)
            time.sleep(2)
            subprocess.run(['netsh', 'interface', 'set', 'interface', 'name=*', 'admin=enabled'], shell=True, check=True)
            logging.info("Network interface reset completed")
        except Exception as e:
            logging.error(f"Error resetting interface: {str(e)}")
    
    def clear_cache(self):
        """Clear network-related caches"""
        try:
            # Clear DNS and ARP caches
            subprocess.run(['ipconfig', '/flushdns'], shell=True, check=True)
            subprocess.run(['arp', '-d', '*'], shell=True, check=True)
            logging.info("Cache clearing completed")
        except Exception as e:
            logging.error(f"Error clearing cache: {str(e)}")
    
    def optimize_tcp(self):
        """Optimize TCP settings"""
        try:
            # Reset TCP/IP stack
            subprocess.run(['netsh', 'int', 'ip', 'reset'], shell=True, check=True)
            subprocess.run(['netsh', 'winsock', 'reset'], shell=True, check=True)
            logging.info("TCP optimization completed")
        except Exception as e:
            logging.error(f"Error optimizing TCP: {str(e)}")
    
    def balance_load(self):
        """Balance network load"""
        try:
            # Reset network adapter binding order
            ps_command = 'Get-NetAdapter | Sort-Object -Property Speed -Descending | Set-NetIPInterface -InterfaceMetric 0'
            subprocess.run(['powershell', '-Command', ps_command], shell=True, check=True)
            logging.info("Load balancing completed")
        except Exception as e:
            logging.error(f"Error balancing load: {str(e)}")
    
    def flush_dns(self):
        """Flush DNS cache"""
        try:
            subprocess.run(['ipconfig', '/flushdns'], shell=True, check=True)
            logging.info("DNS flush completed")
        except Exception as e:
            logging.error(f"Error flushing DNS: {str(e)}")
    
    def do_nothing(self):
        """No action"""
        pass
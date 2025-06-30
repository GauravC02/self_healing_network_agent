import numpy as np
from mininet.log import setLogLevel
import time
import subprocess
from topology.network_simulator import RealNetworkSimulator
class MininetNetworkEnvironment:
    def __init__(self):
        setLogLevel('info')
        self.simulator = RealNetworkSimulator()
        self.net = None
        self.action_space = 8  # Increased action space for real network operations
        self.state_space = 80  # Fixed state vector size
        self.episode_length = 50
        self.current_step = 0
        self.baseline_performance = 0
        
        # Action mapping
        self.actions = {
            0: self.restart_failed_switch,
            1: self.reroute_traffic,
            2: self.adjust_link_bandwidth,
            3: self.restart_host_service,
            4: self.clear_flow_tables,
            5: self.load_balance_traffic,
            6: self.reduce_packet_loss,
            7: self.do_nothing
        }
    
    def reset(self):
        """Reset environment for new episode"""
        if self.net:
            self.simulator.stop_network()
        
        # Create and start new network
        self.net = self.simulator.create_enterprise_topology()
        self.simulator.start_network()
        
        # Wait for network to stabilize
        time.sleep(10)
        
        # Measure baseline performance
        self.baseline_performance = self.simulator.calculate_network_performance()
        
        self.current_step = 0
        return self.simulator.get_network_state_vector()
    
    def step(self, action):
        """Execute one step in the environment"""
        prev_performance = self.simulator.calculate_network_performance()
        prev_state = self.simulator.get_network_state_vector()
        
        # Execute healing action and get immediate reward
        try:
            action_reward = self.actions[action]()
        except Exception as e:
            print(f"Error executing action {action}: {str(e)}")
            action_reward = -5  # Penalty for failed action
        
        # Inject random failures (simulate dynamic network)
        if np.random.random() < 0.15:  # 15% chance of new failure
            failure_types = ['link_failure', 'congestion', 'high_latency', 'packet_loss']
            failure_type = np.random.choice(failure_types)
            self.simulator.inject_network_failure(failure_type)
        
        # Wait for network to react
        time.sleep(3)
        
        # Get new state and performance
        new_state = self.simulator.get_network_state_vector()
        new_performance = self.simulator.calculate_network_performance()
        
        # Calculate reward components
        performance_improvement = new_performance - prev_performance
        state_improvement = np.mean(new_state) - np.mean(prev_state)
        
        # Combine rewards
        reward = 0
        reward += action_reward  # Immediate action reward
        reward += performance_improvement * 2  # Performance improvement reward
        reward += state_improvement * 1  # State improvement reward
        
        # Performance thresholds rewards
        if new_performance > 80:
            reward += 5  # Bonus for high performance
        elif new_performance < 30:
            reward -= 5  # Penalty for very low performance
        
        # Normalize reward to prevent extreme values
        reward = np.clip(reward, -10, 10)
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        info = {
            'performance': new_performance,
            'improvement': performance_improvement,
            'baseline': self.baseline_performance,
            'action_reward': action_reward,
            'state_improvement': state_improvement
        }
        
        return new_state, reward, done, info
    
    def restart_failed_switch(self):
        """Restart failed switches"""
        try:
            # Find failed switches and restart them
            failed_switches = []
            for switch in self.simulator.switches:
                # Check if switch is responding
                result = subprocess.run(["powershell", "-Command", f"Get-NetAdapter | Where-Object {{$_.Name -eq '{switch.name}'}}"], 
                                      capture_output=True, text=True)
                if "Disabled" in result.stdout or result.returncode != 0:
                    failed_switches.append(switch)
                    # Enable the network adapter
                    subprocess.run(["powershell", "-Command", f"Enable-NetAdapter -Name '{switch.name}' -Confirm:$false"])
            
            if failed_switches:
                return 5  # Reward for fixing failed switches
            return 0  # No failed switches found
        except Exception as e:
            logging.error(f"Error in restart_failed_switch: {str(e)}")
            return -2

    def reroute_traffic(self):
        """Reroute traffic to optimize network flow"""
        try:
            # Clear ARP cache
            subprocess.run(["arp", "-d", "*"], capture_output=True)
            
            # Flush DNS cache
            subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
            
            # Reset network interfaces
            subprocess.run(["netsh", "interface", "ip", "delete", "destinationcache"], capture_output=True)
            
            return 3
        except Exception as e:
            logging.error(f"Error in reroute_traffic: {str(e)}")
            return -2

    def adjust_link_bandwidth(self):
        """Adjust link bandwidth settings"""
        try:
            # Get network interfaces
            result = subprocess.run(["powershell", "-Command", "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'}"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Reset network adapter settings
                subprocess.run(["powershell", "-Command", "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | Set-NetAdapterAdvancedProperty -DisplayName 'Speed & Duplex' -DisplayValue 'Auto Negotiation'"])
                return 2
            return 0
        except Exception as e:
            logging.error(f"Error in adjust_link_bandwidth: {str(e)}")
            return -2

    def restart_host_service(self):
        """Restart network-related services"""
        try:
            services = [
                "DNS",
                "Dhcp",
                "NlaSvc",  # Network Location Awareness
                "nsi"      # Network Store Interface Service
            ]
            
            for service in services:
                # Restart each service
                subprocess.run(["powershell", "-Command", f"Restart-Service -Name {service} -Force"], capture_output=True)
            
            return 2
        except Exception as e:
            logging.error(f"Error in restart_host_service: {str(e)}")
            return -2

    def clear_flow_tables(self):
        """Clear network flow tables and caches"""
        try:
            # Reset TCP/IP stack
            subprocess.run(["netsh", "int", "ip", "reset"], capture_output=True)
            subprocess.run(["netsh", "int", "ipv4", "reset"], capture_output=True)
            subprocess.run(["netsh", "int", "ipv6", "reset"], capture_output=True)
            
            # Reset Winsock catalog
            subprocess.run(["netsh", "winsock", "reset"], capture_output=True)
            
            return 3
        except Exception as e:
            logging.error(f"Error in clear_flow_tables: {str(e)}")
            return -2
    def load_balance_traffic(self):
        """Load balance network traffic across available interfaces"""
        try:
            # Get active network interfaces
            result = subprocess.run(["powershell", "-Command", "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'}"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Reset interface metrics to auto
                subprocess.run(["powershell", "-Command", "Get-NetIPInterface | Set-NetIPInterface -AutomaticMetric Enabled"])
                
                # Enable RSS (Receive Side Scaling) on supported adapters
                subprocess.run(["powershell", "-Command", "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | Set-NetAdapterRss -Enabled $True"])
                
                return 4
            return 0
        except Exception as e:
            logging.error(f"Error in load_balance_traffic: {str(e)}")
            return -2

    def reduce_packet_loss(self):
        """Implement measures to reduce packet loss"""
        try:
            # Optimize TCP parameters
            subprocess.run(["netsh", "int", "tcp", "set", "global", "autotuninglevel=normal"], capture_output=True)
            subprocess.run(["netsh", "int", "tcp", "set", "global", "chimney=disabled"], capture_output=True)
            subprocess.run(["netsh", "int", "tcp", "set", "global", "rss=enabled"], capture_output=True)
            
            # Clear DNS cache
            subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
            
            return 3
        except Exception as e:
            logging.error(f"Error in reduce_packet_loss: {str(e)}")
            return -2

    def do_nothing(self):
        """No action taken"""
        return 0
    
    def cleanup(self):
        """Cleanup network resources"""
        if self.simulator:
            self.simulator.stop_network()
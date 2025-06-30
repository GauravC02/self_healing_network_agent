import psutil
import netifaces
import time
import threading
from collections import defaultdict
import subprocess
import logging
import numpy as np
import json
import re
from scapy.all import *

class NetworkMonitor:
    def __init__(self):
        self.interfaces = []
        self.network_stats = defaultdict(dict)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.baseline_stats = None
        self.state_size = 80
        self.last_update_time = 0
        self.update_interval = 1  # seconds
        
        # Initialize logging with debug level
        try:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('network_monitor.log'),
                    logging.StreamHandler()
                ]
            )
            
            # Ensure PowerShell execution policy allows running commands
            cmd = 'Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force'
            subprocess.run(['powershell', '-Command', cmd], capture_output=True)
            
            # Test PowerShell access to network adapters
            test_cmd = 'Get-NetAdapter | Select-Object -First 1'
            result = subprocess.run(['powershell', '-Command', test_cmd], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to access network adapters: {result.stderr}")
            
        except Exception as e:
            logging.error(f"Error initializing network monitor: {str(e)}")
            raise
    
    def initialize(self):
        """Initialize network monitoring by discovering interfaces and collecting baseline statistics"""
        try:
            # Get list of network interfaces using PowerShell
            cmd = 'Get-NetAdapter | Where-Object { $_.Status -eq "Up" } | Select-Object -ExpandProperty InterfaceDescription'
            result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                # Split output into lines and remove empty lines
                active_interfaces = [iface.strip() for iface in result.stdout.split('\n') if iface.strip()]
                if active_interfaces:
                    self.interfaces = active_interfaces
                    logging.info(f"Discovered active network interfaces: {self.interfaces}")
                else:
                    # Fallback to netifaces if no active interfaces found
                    self.interfaces = netifaces.interfaces()
                    logging.warning("No active interfaces found via PowerShell, falling back to netifaces")
            else:
                # Fallback to netifaces if PowerShell command fails
                self.interfaces = netifaces.interfaces()
                logging.warning(f"PowerShell interface detection failed, falling back to netifaces: {result.stderr}")
            
            if not self.interfaces:
                raise RuntimeError("No network interfaces found")
            
            # Initialize baseline statistics with retries
            max_retries = 3
            retry_delay = 2
            baseline_collected = False
            
            # Start monitoring thread first
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_network)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logging.info("Network monitoring started")
            
            # Wait for monitoring thread to start
            time.sleep(1)
            
            # Try to collect baseline statistics
            for attempt in range(max_retries):
                try:
                    # Wait for initial stats to be collected
                    stats = self._collect_network_stats()
                    if stats and any(stats.values()):
                        self.baseline_stats = stats
                        baseline_collected = True
                        logging.info(f"Successfully collected baseline statistics on attempt {attempt + 1}")
                        logging.debug(f"Baseline stats: {self.baseline_stats}")
                        break
                    else:
                        logging.warning(f"Attempt {attempt + 1}/{max_retries}: No valid baseline statistics collected")
                except Exception as e:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    logging.info(f"Waiting {retry_delay} seconds before next retry...")
                    time.sleep(retry_delay)
            
            if not baseline_collected:
                logging.warning("Failed to collect baseline statistics after all retries")
                self.baseline_stats = defaultdict(lambda: {
                    'bandwidth_utilization': 0,
                    'latency': 0,
                    'packet_loss': 0,
                    'error_rate': 0,
                    'throughput': 0
                })
            
            # Verify monitoring thread started successfully
            if not self.monitoring_thread.is_alive():
                raise RuntimeError("Failed to start monitoring thread")
            
            # Log successful initialization
            if baseline_collected:
                baseline_score = self.calculate_performance_score()
                logging.info(f"Network monitoring initialized. Baseline performance: {baseline_score:.2f}%")
            else:
                logging.info("Network monitoring initialized with default baseline values")
            
        except Exception as e:
            logging.error(f"Error initializing network monitoring: {str(e)}")
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=1)
            raise
    
    def start_monitoring(self):
        """Start network monitoring in a separate thread with error handling"""
        try:
            if self.monitoring_active:
                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    logging.warning("Network monitoring is already active")
                    return
                else:
                    logging.warning("Monitoring thread died unexpectedly, restarting")
            
            # Reset monitoring state
            self.monitoring_active = True
            self.last_update_time = 0
            
            # Start new monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitor_network)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            # Verify thread started successfully
            time.sleep(0.1)
            if not self.monitoring_thread.is_alive():
                raise RuntimeError("Failed to start monitoring thread")
            
            logging.info("Network monitoring started successfully")
            
        except Exception as e:
            self.monitoring_active = False
            self.monitoring_thread = None
            logging.error(f"Failed to start network monitoring: {str(e)}")
            raise
    
    def stop_monitoring(self):
        """Stop network monitoring with cleanup"""
        try:
            if not self.monitoring_active:
                logging.warning("Network monitoring is already stopped")
                return
            
            # Signal thread to stop
            self.monitoring_active = False
            
            # Wait for thread to finish with timeout
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                logging.info("Waiting for monitoring thread to stop...")
                self.monitoring_thread.join(timeout=2)
                
                if self.monitoring_thread.is_alive():
                    logging.warning("Monitoring thread did not stop gracefully")
                else:
                    logging.info("Monitoring thread stopped successfully")
            
            # Clear monitoring state
            self.monitoring_thread = None
            self.last_update_time = 0
            
        except Exception as e:
            logging.error(f"Error stopping network monitoring: {str(e)}")
            raise
        finally:
            # Ensure monitoring is marked as inactive
            self.monitoring_active = False
            logging.info("Network monitoring stopped")
    
    def get_network_state(self):
        """Get current network state as a vector for the DQN agent"""
        state = []
        
        for iface in self.interfaces:
            stats = self.network_stats.get(iface, {})
            # Interface metrics
            state.extend([
                stats.get('bandwidth_utilization', 0),
                stats.get('latency', 0),
                stats.get('packet_loss', 0),
                stats.get('error_rate', 0),
                stats.get('throughput', 0)
            ])
        
        # Pad or truncate to fixed size
        if len(state) < self.state_size:
            state.extend([0] * (self.state_size - len(state)))
        else:
            state = state[:self.state_size]
        
        return np.array(state, dtype=np.float32)
    
    def calculate_performance_score(self):
        """Calculate overall network performance score with improved error handling"""
        if not self.network_stats:
            logging.warning("No network statistics available for performance calculation")
            return 0.0
        
        total_score = 0
        num_interfaces = 0
        weights = {
            'bandwidth': 0.3,
            'latency': 0.3,
            'packet_loss': 0.2,
            'error_rate': 0.2
        }
        
        for iface, stats in self.network_stats.items():
            try:
                if not stats or not isinstance(stats, dict):
                    logging.warning(f"Invalid or missing statistics for interface {iface}")
                    continue
                
                # Validate and normalize metrics
                bandwidth_util = min(max(float(stats.get('bandwidth_utilization', 0)), 0), 100)
                latency = min(max(float(stats.get('latency', 1000)), 0), 1000)  # Cap at 1000ms
                packet_loss = min(max(float(stats.get('packet_loss', 0)), 0), 100)
                error_rate = min(max(float(stats.get('error_percentage', 0)), 0), 100)
                
                # Calculate individual metric scores (0-100)
                bandwidth_score = 100 - bandwidth_util  # Lower utilization is better
                latency_score = max(0, 100 - (latency / 10))  # Lower latency is better
                packet_loss_score = 100 - packet_loss  # Lower packet loss is better
                error_score = 100 - error_rate  # Lower error rate is better
                
                # Log individual scores for debugging
                logging.debug(f"Interface {iface} scores:")
                logging.debug(f"  Bandwidth: {bandwidth_score:.2f}")
                logging.debug(f"  Latency: {latency_score:.2f}")
                logging.debug(f"  Packet Loss: {packet_loss_score:.2f}")
                logging.debug(f"  Error Rate: {error_score:.2f}")
                
                # Calculate weighted average for interface
                interface_score = (
                    bandwidth_score * weights['bandwidth'] +
                    latency_score * weights['latency'] +
                    packet_loss_score * weights['packet_loss'] +
                    error_score * weights['error_rate']
                )
                
                logging.debug(f"Interface {iface} overall score: {interface_score:.2f}")
                
                total_score += interface_score
                num_interfaces += 1
                
            except Exception as e:
                logging.error(f"Error calculating performance score for interface {iface}: {str(e)}")
                continue
        
        if num_interfaces == 0:
            logging.warning("No valid interfaces found for performance calculation")
            return 0.0
        
        final_score = total_score / num_interfaces
        logging.info(f"Overall network performance score: {final_score:.2f}%")
        return final_score
            # Removed duplicate code block as it was causing indentation issues
        
        # Return average score, minimum 1%
        return max(1, np.mean(scores) if scores else 1)
    
    def _monitor_network(self):
        """Monitor network statistics in real-time with rate limiting and error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        error_delay = 2  # seconds
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                # Rate limiting: only update if enough time has passed
                if current_time - self.last_update_time >= self.update_interval:
                    current_stats = self._collect_network_stats()
                    
                    if any(current_stats.values()):
                        self.network_stats.update(current_stats)
                        self.last_update_time = current_time
                        consecutive_errors = 0  # Reset error counter on success
                        
                        # Log successful update
                        logging.debug("Network statistics updated successfully")
                    else:
                        logging.warning("No valid network statistics collected")
                        consecutive_errors += 1
                else:
                    # Sleep for the remaining time until next update
                    time.sleep(max(0, self.update_interval - (current_time - self.last_update_time)))
                    continue
                
            except Exception as e:
                logging.error(f"Error in network monitoring: {str(e)}")
                consecutive_errors += 1
                
            # Handle consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                logging.error(f"Multiple consecutive errors ({consecutive_errors}). Increasing delay.")
                time.sleep(error_delay)
                error_delay = min(error_delay * 2, 10)  # Exponential backoff, max 10 seconds
            else:
                time.sleep(self.update_interval)
    
    def _get_interface_speed(self, iface):
        """Get interface speed in MB/s"""
        try:
            # Use PowerShell to get interface speed with error handling
            cmd = f'Get-NetAdapter -Name "{iface}" | Select-Object -ExpandProperty LinkSpeed'
            result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                speed_str = result.stdout.strip()
                try:
                    value = float(speed_str.split()[0])
                    unit = speed_str.split()[1].lower()
                    
                    if 'gbps' in unit:
                        return value * 1024  # Convert Gbps to MBps
                    elif 'mbps' in unit:
                        return value  # Already in MBps
                    else:
                        logging.warning(f"Unknown speed unit for interface {iface}: {unit}")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse speed value for interface {iface}: {speed_str}")
            else:
                logging.warning(f"Failed to get speed for interface {iface}. Using default speed.")
            
        except subprocess.TimeoutExpired:
            logging.warning(f"Timeout getting speed for interface {iface}")
        except Exception as e:
            logging.error(f"Error getting interface speed for {iface}: {str(e)}")
        
        # Default to 1 Gbps = 1024 MBps
        return 1024

    def _measure_packet_loss(self, iface):
        """Measure packet loss rate"""
        try:
            net_io = psutil.net_io_counters(pernic=True).get(iface)
            if net_io:
                total_packets = net_io.packets_sent + net_io.packets_recv
                if total_packets > 0:
                    return (net_io.dropin + net_io.dropout) / total_packets
        except Exception as e:
            logging.error(f"Error measuring packet loss: {str(e)}")
        return 0.0

    def _measure_latency(self, iface):
        """Measure network latency with improved accuracy and timeout handling"""
        try:
            # First try to get gateway specific to the interface
            cmd = f'Get-NetRoute -InterfaceAlias "{iface}" | Where-Object {{ $_.DestinationPrefix -eq "0.0.0.0/0" }} | Select-Object -ExpandProperty NextHop'
            result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0 and result.stdout.strip():
                gateway = result.stdout.strip()
            else:
                # Fallback to default gateway
                gateway = self._get_default_gateway()
            
            if not gateway:
                logging.warning(f"No gateway found for interface {iface}")
                return 1000
            
            # Use PowerShell Test-Connection for more reliable ping measurements
            ps_cmd = f'$result = Test-Connection -ComputerName {gateway} -Count 3 -ErrorAction SilentlyContinue; if ($result) {{ ($result | Measure-Object -Property ResponseTime -Average).Average }} else {{ 1000 }}'
            result = subprocess.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    latency = float(result.stdout.strip())
                    # Cap latency at 1000ms and ensure it's not negative
                    return max(0, min(latency, 1000))
                except ValueError:
                    logging.warning(f"Invalid latency value received: {result.stdout.strip()}")
            else:
                logging.warning(f"No response from gateway {gateway} for interface {iface}")
                
        except subprocess.TimeoutExpired:
            logging.warning(f"Latency measurement timed out for interface {iface}")
        except Exception as e:
            logging.error(f"Error measuring latency for interface {iface}: {str(e)}")
        
        return 1000  # Default to high latency for failed measurements

def _collect_network_stats(self):
    """
    Collect current network statistics for all interfaces using PowerShell.

    Returns:
    dict: A dictionary containing interface names as keys and their statistics as values.
    """
    stats = {}

    try:
        cmd = '''
        $adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" }
        $results = @()
        foreach ($adapter in $adapters) {
            try {
                $stats = Get-NetAdapterStatistics -Name $adapter.Name -ErrorAction Stop
                $speed = $adapter.LinkSpeed
                $properties = @{
                    "Name" = $adapter.InterfaceDescription
                    "BytesSent" = [long]$stats.SentBytes
                    "BytesReceived" = [long]$stats.ReceivedBytes
                    "PacketsSent" = [long]$stats.SentPackets
                    "PacketsReceived" = [long]$stats.ReceivedPackets
                    "ErrorsIn" = [long]$stats.ReceivedErrors
                    "ErrorsOut" = [long]$stats.OutboundErrors
                    "DropsIn" = [long]$stats.ReceivedDiscardedPackets
                    "DropsOut" = [long]$stats.OutboundDiscardedPackets
                    "Speed" = $speed
                }
                $results += $properties
            } catch {
                Write-Error "Failed to get statistics for adapter $($adapter.Name): $_"
            }
        }
        $results | ConvertTo-Json
        '''
        result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True)

        if result.returncode == 0 and result.stdout.strip():
            try:
                adapter_stats = json.loads(result.stdout)
                if not isinstance(adapter_stats, list):
                    adapter_stats = [adapter_stats]

                for adapter_data in adapter_stats:
                    if not adapter_data:
                        self.logger.warning("Empty adapter data")
                        continue

                    iface = adapter_data.get('Name')
                    if not iface:
                        self.logger.warning("Missing interface name in adapter data")
                        continue

                    time_delta = time.time() - self.network_stats.get(iface, {}).get('last_time', time.time())

                    def calc_delta(current, previous):
                        if current >= previous:
                            return current - previous
                        elif previous - current > 1e12:
                            self.logger.warning(f"Counter reset detected: {previous} -> {current}")
                            return current
                        else:
                            return 0

                    try:
                        bytes_sent = int(adapter_data.get('BytesSent', 0))
                        bytes_recv = int(adapter_data.get('BytesReceived', 0))
                        last_bytes_sent = int(self.network_stats.get(iface, {}).get('last_bytes_sent', 0))
                        last_bytes_recv = int(self.network_stats.get(iface, {}).get('last_bytes_received', 0))
                        bytes_sent_delta = calc_delta(bytes_sent, last_bytes_sent)
                        bytes_recv_delta = calc_delta(bytes_recv, last_bytes_recv)
                    except Exception as e:
                        self.logger.error(f"Error calculating byte deltas for {iface}: {e}")
                        bytes_sent_delta = bytes_recv_delta = 0

                    try:
                        packets_sent = int(adapter_data.get('PacketsSent', 0))
                        packets_recv = int(adapter_data.get('PacketsReceived', 0))
                        last_packets_sent = int(self.network_stats.get(iface, {}).get('last_packets_sent', 0))
                        last_packets_recv = int(self.network_stats.get(iface, {}).get('last_packets_received', 0))
                        packets_sent_delta = calc_delta(packets_sent, last_packets_sent)
                        packets_recv_delta = calc_delta(packets_recv, last_packets_recv)
                    except Exception as e:
                        self.logger.error(f"Error calculating packet deltas for {iface}: {e}")
                        packets_sent_delta = packets_recv_delta = 0

                    try:
                        errors_in = int(adapter_data.get('ErrorsIn', 0))
                        errors_out = int(adapter_data.get('ErrorsOut', 0))
                        last_errors_in = int(self.network_stats.get(iface, {}).get('last_errors_in', 0))
                        last_errors_out = int(self.network_stats.get(iface, {}).get('last_errors_out', 0))
                        errors_in_delta = calc_delta(errors_in, last_errors_in)
                        errors_out_delta = calc_delta(errors_out, last_errors_out)
                    except Exception as e:
                        self.logger.error(f"Error calculating error deltas for {iface}: {e}")
                        errors_in_delta = errors_out_delta = 0

                    try:
                        drops_in = int(adapter_data.get('DropsIn', 0))
                        drops_out = int(adapter_data.get('DropsOut', 0))
                        last_drops_in = int(self.network_stats.get(iface, {}).get('last_drops_in', 0))
                        last_drops_out = int(self.network_stats.get(iface, {}).get('last_drops_out', 0))
                        drops_in_delta = calc_delta(drops_in, last_drops_in)
                        drops_out_delta = calc_delta(drops_out, last_drops_out)
                    except Exception as e:
                        self.logger.error(f"Error calculating drop deltas for {iface}: {e}")
                        drops_in_delta = drops_out_delta = 0

                    try:
                        throughput = (bytes_sent_delta + bytes_recv_delta) / time_delta if time_delta > 0 else 0
                        packet_rate = (packets_sent_delta + packets_recv_delta) / time_delta if time_delta > 0 else 0
                        error_rate = (errors_in_delta + errors_out_delta) / time_delta if time_delta > 0 else 0
                        drop_rate = (drops_in_delta + drops_out_delta) / time_delta if time_delta > 0 else 0
                    except Exception as e:
                        self.logger.error(f"Error calculating rates for {iface}: {e}")
                        throughput = packet_rate = error_rate = drop_rate = 0

                    try:
                        speed_str = adapter_data.get('Speed', '')
                        speed_match = re.match(r'([\d.]+)\s*(Gbps|Mbps)', speed_str)
                        if speed_match:
                            speed_val = float(speed_match.group(1))
                            unit = speed_match.group(2)
                            interface_speed = speed_val * (1000 if unit == 'Gbps' else 1)
                        else:
                            self.logger.warning(f"Could not parse speed string: '{speed_str}'")
                            interface_speed = 1000
                    except Exception as e:
                        self.logger.error(f"Error parsing interface speed for {iface}: {e}")
                        interface_speed = 1000

                    try:
                        utilization = (throughput * 8) / (interface_speed * 1024 * 1024) * 100 if interface_speed else 0
                        packet_loss = (drop_rate / packet_rate) * 100 if packet_rate > 0 else 0
                        error_percentage = (error_rate / packet_rate) * 100 if packet_rate > 0 else 0

                        try:
                            latency = self._measure_latency(iface)
                        except Exception as e:
                            self.logger.warning(f"Latency check failed for {iface}: {e}")
                            latency = float('inf')

                        stats[iface] = {
                            'throughput': max(0, throughput),
                            'packet_rate': max(0, packet_rate),
                            'error_rate': max(0, error_rate),
                            'drop_rate': max(0, drop_rate),
                            'bandwidth_utilization': min(100, max(0, utilization)),
                            'packet_loss': min(100, max(0, packet_loss)),
                            'error_percentage': min(100, max(0, error_percentage)),
                            'latency': max(0, latency)
                        }
                    except Exception as e:
                        self.logger.error(f"Metric calculation failed for {iface}: {e}")
                        stats[iface] = {
                            'throughput': 0,
                            'packet_rate': 0,
                            'error_rate': 0,
                            'drop_rate': 0,
                            'bandwidth_utilization': 0,
                            'packet_loss': 0,
                            'error_percentage': 0,
                            'latency': float('inf')
                        }

                    self.network_stats[iface] = {
                        'last_time': time.time(),
                        'last_bytes_sent': bytes_sent,
                        'last_bytes_received': bytes_recv,
                        'last_packets_sent': packets_sent,
                        'last_packets_received': packets_recv,
                        'last_errors_in': errors_in,
                        'last_errors_out': errors_out,
                        'last_drops_in': drops_in,
                        'last_drops_out': drops_out
                    }

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error processing network stats: {e}")
        else:
            self.logger.error(f"PowerShell error: {result.stderr.strip()}")

    except Exception as e:
        self.logger.error(f"Exception in _collect_network_stats: {e}")

    return stats



def _get_interface_speed(self, iface):
    """Get interface speed in MB/s."""
    try:
        cmd = "wmic NIC where NetEnabled=true get Name, Speed"
        output = subprocess.check_output(cmd, shell=True).decode(errors='ignore')
        for line in output.splitlines():
            if iface in line:
                try:
                    speed = int(line.strip().split()[-1])
                    return speed / 8 / 1024 / 1024  # Convert bits to MB/s
                except ValueError:
                    self.logger.warning(f"Failed to parse speed from line: {line}")
                    return 1000  # Default to 1 Gbps
        self.logger.warning(f"Interface '{iface}' not found in WMIC output.")
        return 1000
    except Exception as e:
        self.logger.error(f"Error getting interface speed for {iface}: {e}")
        return 1000


def _measure_latency(self, iface):
    """Measure network latency by pinging the default gateway."""
    try:
        gateways = netifaces.gateways()
        if 'default' in gateways and netifaces.AF_INET in gateways['default']:
            gateway_ip = gateways['default'][netifaces.AF_INET][0]
            cmd = f"ping -n 3 {gateway_ip}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Average' in line:
                        try:
                            latency_str = line.split('=')[-1].strip().replace("ms", "")
                            return float(latency_str)
                        except ValueError:
                            self.logger.warning(f"Failed to parse latency from line: {line}")
        return 0
    except Exception as e:
        self.logger.error(f"Error measuring latency on {iface}: {e}")
        return 0


def _measure_packet_loss(self, iface):
    """Measure packet loss rate to the default gateway."""
    try:
        gateways = netifaces.gateways()
        if 'default' in gateways and netifaces.AF_INET in gateways['default']:
            gateway_ip = gateways['default'][netifaces.AF_INET][0]
            cmd = f"ping -n 10 {gateway_ip}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Lost' in line and '(' in line:
                        try:
                            loss_percent = int(line.split('(')[1].split('%')[0])
                            return loss_percent / 100
                        except (ValueError, IndexError):
                            self.logger.warning(f"Failed to parse packet loss from line: {line}")
        return 0
    except Exception as e:
        self.logger.error(f"Error measuring packet loss on {iface}: {e}")
        return 0

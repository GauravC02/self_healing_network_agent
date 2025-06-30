from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import subprocess
import threading
from collections import defaultdict
import psutil
import numpy as np
class RealNetworkSimulator:
    def __init__(self):
        self.net = Mininet(controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633))
        self.topology_type = 'tree'
        self.hosts = []
        self.switches = []
        self.links = []
        self.network_stats = defaultdict(dict)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.baseline_stats = None
    
    def get_network_state_vector(self):
        """Generate state vector for DQN agent"""
        state = []
        
        # Link statistics
        for link in self.links:
            stats = self.network_stats.get(link, {})
            state.extend([
                stats.get('bandwidth_utilization', 0),
                stats.get('latency', 0),
                stats.get('packet_loss', 0)
            ])
        
        # Switch statistics
        for switch in self.switches:
            stats = self.network_stats.get(switch, {})
            state.extend([
                stats.get('flow_count', 0),
                stats.get('packet_count', 0),
                stats.get('error_count', 0)
            ])
        
        # Pad or truncate to fixed size
        target_size = 80  # Must match state_space in MininetNetworkEnvironment
        if len(state) < target_size:
            state.extend([0] * (target_size - len(state)))
        else:
            state = state[:target_size]
        
        return np.array(state, dtype=np.float32)
    
    def calculate_network_performance(self):
        """Calculate overall network performance score"""
        if not self.network_stats:
            return 0
        
        scores = []
        
        # Link performance
        for link in self.links:
            stats = self.network_stats.get(link, {})
            if stats:
                # Higher score for better performance
                link_score = 100
                link_score -= stats.get('packet_loss', 0) * 100  # Reduce score based on packet loss
                link_score -= min(stats.get('latency', 0) / 100, 50)  # Reduce score based on latency
                link_score -= stats.get('bandwidth_utilization', 0) / 2  # Reduce score for high utilization
                scores.append(max(0, link_score))
        
        # Switch performance
        for switch in self.switches:
            stats = self.network_stats.get(switch, {})
            if stats:
                switch_score = 100
                switch_score -= min(stats.get('error_count', 0) * 10, 50)
                scores.append(max(0, switch_score))
        
        return np.mean(scores) if scores else 0
    
    def inject_network_failure(self, failure_type):
        """Inject artificial network failures for training"""
        if failure_type == 'link_failure':
            # Simulate link failure by increasing packet loss
            if self.links:
                link = np.random.choice(self.links)
                self.network_stats[link]['packet_loss'] = np.random.uniform(0.3, 0.8)
        
        elif failure_type == 'congestion':
            # Simulate congestion by increasing bandwidth utilization
            if self.links:
                link = np.random.choice(self.links)
                self.network_stats[link]['bandwidth_utilization'] = np.random.uniform(0.8, 1.0)
        
        elif failure_type == 'high_latency':
            # Simulate high latency
            if self.links:
                link = np.random.choice(self.links)
                self.network_stats[link]['latency'] = np.random.uniform(50, 200)
        
        elif failure_type == 'packet_loss':
            # Simulate packet loss on multiple links
            if self.links:
                affected_links = np.random.choice(self.links, size=min(3, len(self.links)), replace=False)
                for link in affected_links:
                    self.network_stats[link]['packet_loss'] = np.random.uniform(0.1, 0.4)

    def create_enterprise_topology(self):
        """Create a realistic enterprise network topology"""
        info('*** Creating enterprise network topology\n')

        self.net = Mininet(
            controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
            switch=OVSKernelSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )

        # Add controller
        c0 = self.net.addController('c0')

        # Core switches (backbone)
        core1 = self.net.addSwitch('s1', cls=OVSKernelSwitch, protocols='OpenFlow13')
        core2 = self.net.addSwitch('s2', cls=OVSKernelSwitch, protocols='OpenFlow13')

        # Distribution layer switches
        dist1 = self.net.addSwitch('s3', cls=OVSKernelSwitch, protocols='OpenFlow13')
        dist2 = self.net.addSwitch('s4', cls=OVSKernelSwitch, protocols='OpenFlow13')
        dist3 = self.net.addSwitch('s5', cls=OVSKernelSwitch, protocols='OpenFlow13')

        # Access layer switches
        access1 = self.net.addSwitch('s6', cls=OVSKernelSwitch, protocols='OpenFlow13')
        access2 = self.net.addSwitch('s7', cls=OVSKernelSwitch, protocols='OpenFlow13')
        access3 = self.net.addSwitch('s8', cls=OVSKernelSwitch, protocols='OpenFlow13')
        access4 = self.net.addSwitch('s9', cls=OVSKernelSwitch, protocols='OpenFlow13')

        self.switches = [core1, core2, dist1, dist2, dist3, access1, access2, access3, access4]

        # Add hosts
        web1 = self.net.addHost('web1', ip='10.0.1.10/24')
        web2 = self.net.addHost('web2', ip='10.0.1.11/24')

        db1 = self.net.addHost('db1', ip='10.0.2.10/24')
        db2 = self.net.addHost('db2', ip='10.0.2.11/24')

        pc1 = self.net.addHost('pc1', ip='10.0.3.10/24')
        pc2 = self.net.addHost('pc2', ip='10.0.3.11/24')
        pc3 = self.net.addHost('pc3', ip='10.0.3.12/24')
        pc4 = self.net.addHost('pc4', ip='10.0.3.13/24')

        monitor = self.net.addHost('monitor', ip='10.0.4.10/24')

        self.hosts = [web1, web2, db1, db2, pc1, pc2, pc3, pc4, monitor]

        # Create links with bandwidth and latency
        self.net.addLink(core1, core2, bw=1000, delay='1ms', loss=0.1)

        self.net.addLink(core1, dist1, bw=1000, delay='2ms', loss=0.1)
        self.net.addLink(core1, dist2, bw=1000, delay='2ms', loss=0.1)
        self.net.addLink(core2, dist2, bw=1000, delay='2ms', loss=0.1)
        self.net.addLink(core2, dist3, bw=1000, delay='2ms', loss=0.1)

        self.net.addLink(dist1, access1, bw=100, delay='5ms', loss=0.2)
        self.net.addLink(dist1, access2, bw=100, delay='5ms', loss=0.2)
        self.net.addLink(dist2, access2, bw=100, delay='5ms', loss=0.2)
        self.net.addLink(dist2, access3, bw=100, delay='5ms', loss=0.2)
        self.net.addLink(dist3, access3, bw=100, delay='5ms', loss=0.2)
        self.net.addLink(dist3, access4, bw=100, delay='5ms', loss=0.2)

        self.net.addLink(web1, access1, bw=100, delay='1ms')
        self.net.addLink(web2, access1, bw=100, delay='1ms')
        self.net.addLink(db1, access2, bw=100, delay='1ms')
        self.net.addLink(db2, access2, bw=100, delay='1ms')
        self.net.addLink(pc1, access3, bw=10, delay='1ms')
        self.net.addLink(pc2, access3, bw=10, delay='1ms')
        self.net.addLink(pc3, access4, bw=10, delay='1ms')
        self.net.addLink(pc4, access4, bw=10, delay='1ms')
        self.net.addLink(monitor, access1, bw=100, delay='1ms')

        return self.net

    def start_network(self):
        info('*** Starting network\n')
        self.net.start()

        for host in self.hosts:
            # Configure hosts
            host.cmd('sysctl -w net.ipv4.tcp_congestion_control=cubic')
        
        # Start monitoring
        self.start_monitoring()
    
    def stop_network(self):
        """Stop the network and monitoring"""
        self.stop_monitoring()
        if self.net:
            self.net.stop()
    
    def start_monitoring(self):
        """Start network monitoring in a separate thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_network)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
    
    def _monitor_network(self):
        """Monitor network statistics in real-time"""
        while self.monitoring_active:
            try:
                # Monitor links
                for link in self.links:
                    if hasattr(link, 'intf1') and hasattr(link, 'intf2'):
                        # Measure bandwidth utilization
                        stats1 = psutil.net_io_counters(pernic=True).get(link.intf1.name, None)
                        stats2 = psutil.net_io_counters(pernic=True).get(link.intf2.name, None)
                        
                        if stats1 and stats2:
                            bytes_sent = stats1.bytes_sent + stats2.bytes_sent
                            bytes_recv = stats1.bytes_recv + stats2.bytes_recv
                            
                            # Calculate utilization based on link capacity
                            capacity = getattr(link, 'bw', 10) * 1000000 / 8  # Convert Mbps to bytes/s
                            utilization = (bytes_sent + bytes_recv) / capacity
                            self.network_stats[link]['bandwidth_utilization'] = min(1.0, utilization)
                        
                        # Measure latency
                        latency = self._measure_latency(link.intf1.name, link.intf2.name)
                        if latency is not None:
                            self.network_stats[link]['latency'] = latency
                        
                        # Measure packet loss
                        loss = self._measure_packet_loss(link.intf1.name, link.intf2.name)
                        if loss is not None:
                            self.network_stats[link]['packet_loss'] = loss
                
                # Monitor switches
                for switch in self.switches:
                    # Get flow statistics
                    flow_stats = self._get_switch_flow_stats(switch)
                    if flow_stats:
                        self.network_stats[switch].update(flow_stats)
                
                time.sleep(1)  # Update interval
            
            except Exception as e:
                print(f"Error in network monitoring: {str(e)}")
                time.sleep(1)
    
    def _measure_latency(self, intf1, intf2):
        """Measure latency between two interfaces"""
        try:
            result = subprocess.run(
                ['ping', '-c', '3', '-q', intf2],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Extract average latency from ping output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'avg' in line:
                        latency = float(line.split('/')[-3])
                        return latency
            return None
        except:
            return None
    
    def _measure_packet_loss(self, intf1, intf2):
        """Measure packet loss between two interfaces"""
        try:
            result = subprocess.run(
                ['ping', '-c', '10', '-q', intf2],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Extract packet loss percentage from ping output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'packet loss' in line:
                        loss = float(line.split('%')[0].split(' ')[-1]) / 100
                        return loss
            return None
        except:
            return None
    
    def _get_switch_flow_stats(self, switch):
        """Get flow statistics from a switch"""
        try:
            result = subprocess.run(
                ['ovs-ofctl', 'dump-flows', switch.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                flows = result.stdout.strip().split('\n')
                return {
                    'flow_count': len(flows),
                    'packet_count': sum(int(flow.split('n_packets=')[1].split(',')[0]) for flow in flows if 'n_packets=' in flow),
                    'error_count': 0  # Initialize error count
                }
            return None
        except:
            return None
            host.cmd('sysctl net.ipv4.ip_forward=1')

        self.start_network_monitoring()
        info('*** Network started successfully\n')

    def start_network_monitoring(self):
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self._monitor_network_stats)
        monitor_thread.daemon = True
        monitor_thread.start()

    def _monitor_network_stats(self):
        while self.monitoring_active:
            try:
                for switch in self.switches:
                    stats = self._get_switch_stats(switch.name)
                    self.network_stats[switch.name] = stats

                for host in self.hosts:
                    stats = self._get_host_stats(host.name)
                    self.network_stats[host.name] = stats

                time.sleep(5)

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)

    def _get_switch_stats(self, switch_name):
        try:
            cmd = f"sudo ovs-ofctl -O OpenFlow13 dump-port-stats {switch_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            stats = {
                'timestamp': time.time(),
                'ports': {},
                'flows': 0,
                'packet_count': 0,
                'byte_count': 0
            }

            for line in result.stdout.split('\n'):
                if 'port' in line and 'rx' in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        port_num = parts[1].rstrip(':')
                        rx_packets = int(parts[2].split('=')[1].rstrip(','))
                        tx_packets = int(parts[6].split('=')[1].rstrip(','))

                        stats['ports'][port_num] = {
                            'rx_packets': rx_packets,
                            'tx_packets': tx_packets,
                            'total_packets': rx_packets + tx_packets
                        }

                        stats['packet_count'] += rx_packets + tx_packets

            cmd = f"sudo ovs-ofctl -O OpenFlow13 dump-flows {switch_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            stats['flows'] = len(result.stdout.split('\n')) - 1

            return stats

        except Exception as e:
            return {'error': str(e), 'timestamp': time.time()}

    def _get_host_stats(self, host_name):
        try:
            cmd = f"sudo ip netns exec {host_name} cat /proc/net/dev"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            stats = {
                'timestamp': time.time(),
                'interfaces': {},
                'cpu_usage': 0,
                'memory_usage': 0
            }

            for line in result.stdout.split('\n')[2:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 17:
                        iface = parts[0].rstrip(':')
                        rx_bytes = int(parts[1])
                        tx_bytes = int(parts[9])

                        stats['interfaces'][iface] = {
                            'rx_bytes': rx_bytes,
                            'tx_bytes': tx_bytes,
                            'total_bytes': rx_bytes + tx_bytes
                        }

            stats['cpu_usage'] = psutil.cpu_percent()
            stats['memory_usage'] = psutil.virtual_memory().percent

            return stats

        except Exception as e:
            return {'error': str(e), 'timestamp': time.time()}

    def inject_network_failure(self, failure_type, target=None):
        info(f'*** Injecting failure: {failure_type}\n')

        if failure_type == 'link_failure':
            if not target:
                link_pairs = [
                    ('s1', 's2'), ('s1', 's3'), ('s2', 's4'),
                    ('s3', 's6'), ('s4', 's7'), ('s5', 's8')
                ]
                target = link_pairs[np.random.randint(len(link_pairs))]

            self.net.configLinkStatus(target[0], target[1], 'down')
            return f"Link {target[0]}-{target[1]} failed"

        elif failure_type == 'switch_failure':
            if not target:
                target = np.random.choice(['s6', 's7', 's8', 's9'])

            switch = self.net.get(target)
            switch.stop()
            return f"Switch {target} failed"

        elif failure_type == 'congestion':
            if not target:
                target = np.random.choice(['s3', 's4', 's5'])

            cmd = f"sudo ovs-vsctl set port {target} qos=@newqos -- --id=@newqos create qos type=linux-htb other-config:max-rate=10000000"
            subprocess.run(cmd, shell=True)
            return f"Congestion created at {target}"

        elif failure_type == 'high_latency':
            if not target:
                target = ('s1', 's3')

            self.net.configLinkStatus(target[0], target[1], 'down')
            time.sleep(0.1)
            self.net.configLinkStatus(target[0], target[1], 'up')
            return f"High latency injected on {target[0]}-{target[1]}"

        elif failure_type == 'packet_loss':
            if not target:
                target = 's2'

            cmd = f"sudo tc qdisc add dev {target}-eth1 root netem loss 10%"
            subprocess.run(cmd, shell=True)
            return f"Packet loss injected at {target}"

    def get_network_state_vector(self):
        state = []

        for switch_name in sorted([s.name for s in self.switches]):
            if switch_name in self.network_stats:
                stats = self.network_stats[switch_name]
                state.extend([
                    stats.get('packet_count', 0) / 10000.0,
                    stats.get('flows', 0) / 100.0,
                    len(stats.get('ports', {})) / 10.0,
                    1.0 if 'error' not in stats else 0.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        for host_name in sorted([h.name for h in self.hosts]):
            if host_name in self.network_stats:
                stats = self.network_stats[host_name]
                total_bytes = sum(iface.get('total_bytes', 0)
                                  for iface in stats.get('interfaces', {}).values())
                state.extend([
                    total_bytes / 1000000.0,
                    stats.get('cpu_usage', 0) / 100.0,
                    stats.get('memory_usage', 0) / 100.0,
                    1.0 if 'error' not in stats else 0.0
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])

        target_size = 80
        if len(state) < target_size:
            state.extend([0.0] * (target_size - len(state)))
        elif len(state) > target_size:
            state = state[:target_size]

        return np.array(state)

    def calculate_network_performance(self):
        try:
            web1 = self.net.get('web1')
            db1 = self.net.get('db1')
            pc1 = self.net.get('pc1')

            connectivity_score = 0
            latency_score = 0

            result = web1.cmd('ping -c 3 10.0.2.10')
            if '0% packet loss' in result:
                connectivity_score += 25
                if 'avg' in result:
                    avg_latency = float(result.split('avg')[1].split('/')[0])
                    latency_score += max(0, 50 - avg_latency)

            result = pc1.cmd('ping -c 3 10.0.1.10')
            if '0% packet loss' in result:
                connectivity_score += 25
                if 'avg' in result:
                    avg_latency = float(result.split('avg')[1].split('/')[0])
                    latency_score += max(0, 50 - avg_latency)

            bandwidth_score = 0
            try:
                web1.cmd('iperf3 -s -D')
                time.sleep(1)
                result = pc1.cmd('iperf3 -c 10.0.1.10 -t 5 -f M')
                if 'Mbits/sec' in result:
                    bandwidth = float(result.split('Mbits/sec')[0].split()[-1])
                    bandwidth_score = min(50, bandwidth)
                web1.cmd('pkill iperf3')
            except Exception as e:
                print(f"Bandwidth test error: {e}")

            total_score = connectivity_score + latency_score + bandwidth_score
            return min(100, total_score)

        except Exception as e:
            print(f"Performance calculation error: {e}")
            return 0

    def stop_network(self):
        self.monitoring_active = False
        if self.net:
            self.net.stop()
        info('*** Network stopped\n')




# Self-Healing Network Agent

An intelligent network agent that uses Deep Q-Learning (DQN) to autonomously identify and fix network issues, optimize performance, and maintain network health in real-time.

## Features

- **Autonomous Network Monitoring**: Continuously monitors network performance, latency, bandwidth, and packet loss
- **Intelligent Problem Detection**: Uses DQN to identify network bottlenecks, failures, and performance issues
- **Automatic Healing Actions**: Implements autonomous healing actions including:
  - Traffic rerouting
  - Link bandwidth adjustment
  - Load balancing
  - Service restart
  - Flow table optimization
  - Packet loss reduction
- **Real-time Performance Optimization**: Maintains optimal network performance through continuous learning and adaptation
- **Detailed Logging**: Comprehensive logging of network states, healing actions, and performance metrics

## Prerequisites

- Python 3.8 or higher
- Mininet
- OpenFlow-compatible switches (OVS)
- CUDA-capable GPU (recommended for faster training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/self-healing-network-agent.git
cd self-healing-network-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Mininet (Linux only):
```bash
sudo apt-get install mininet
```

## Usage

### Training the Agent

1. Start the training process:
```bash
python train.py
```

The training script will:
- Create a simulated network environment
- Train the DQN agent through multiple episodes
- Save the best performing model
- Record training metrics and performance data

### Deploying the Agent

1. Deploy the trained agent:
```bash
python deploy.py
```

The deployment script will:
- Load the trained model
- Monitor network performance in real-time
- Automatically detect and fix network issues
- Log all healing actions and performance metrics

## Network Environment

The agent operates in a realistic enterprise network topology featuring:
- Core layer (backbone switches)
- Distribution layer
- Access layer
- Various host types (web servers, databases, client machines)
- Configurable link properties (bandwidth, latency, loss)

## Monitoring and Metrics

The agent tracks various network metrics including:
- Link utilization
- Latency
- Packet loss
- Throughput
- Service availability
- Overall network performance

## Healing Actions

The agent can perform the following healing actions:
1. Restart failed switches
2. Reroute traffic around congested or failed links
3. Adjust link bandwidth allocations
4. Restart problematic network services
5. Clear and optimize flow tables
6. Implement load balancing
7. Reduce packet loss through path optimization

## Logs and Data

All monitoring data and healing actions are logged in:
- `network_healing.log`: Detailed event and action logs
- `monitoring_data_[timestamp].json`: Performance metrics and healing events
- `models/`: Saved model checkpoints and training metrics

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mininet team for the network emulation platform
- PyTorch team for the deep learning framework
- OpenFlow and OVS communities
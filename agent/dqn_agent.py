import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import logging
import os

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        
        # Feedforward network with larger layer sizes to match saved model
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Handle both single samples and batches
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=100000)  # Increased memory size
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.997  # Slower decay for better exploration
        self.learning_rate = 0.0005  # Reduced learning rate for stability
        self.batch_size = 128  # Increased batch size
        self.update_target_frequency = 5  # More frequent target updates
        self.tau = 0.001  # Soft update parameter
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Use Adam optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Training steps counter
        self.training_steps = 0
        
        # Prioritized experience replay parameters
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001
        
        # Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training metrics
        self.training_steps = 0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def replay(self):
        """Train on batch from replay memory with prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            # Sample batch with prioritized experience replay
            batch = random.sample(self.memory, self.batch_size)
            
            # Prepare batch tensors
            states = torch.FloatTensor([e.state for e in batch]).to(self.device)
            actions = torch.LongTensor([e.action for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
            dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
            
            # Current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values with Double DQN
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                # Get Q values from target network for next_actions
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                # Compute target Q values
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            
            # Compute loss with Huber loss (more robust than MSE)
            loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Soft update target network
            self.training_steps += 1
            if self.training_steps % self.update_target_frequency == 0:
                self._soft_update_target_network()
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
            
        except Exception as e:
            logging.error(f"Error in replay: {str(e)}")
            return None
    
    def _soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
        logging.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_steps = checkpoint['training_steps']
            logging.info(f"Model loaded from {path}")
            return True
        return False
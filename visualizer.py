import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class NetworkVisualizer:
    def __init__(self):
        plt.ion()  # Enable interactive mode
        
        # Create figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Network Monitoring Visualization', fontsize=16)
        
        # Initialize data lists
        self.performance_data = []
        self.reward_data = []
        self.timestamps = []
        
        # Initialize lines
        self.performance_line, = self.ax1.plot([], [], 'b-', label='Performance')
        self.reward_line, = self.ax2.plot([], [], 'g-', label='Reward')
        
        # Setup axes
        self.ax1.set_title('Network Performance')
        self.ax1.set_ylabel('Performance (%)')
        self.ax1.set_ylim(0, 100)
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_title('Healing Action Rewards')
        self.ax2.set_ylabel('Reward')
        self.ax2.set_ylim(-1, 1)
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        self.fig.subplots_adjust(top=0.9)
        
    def update(self, performance, reward):
        # Append new data
        self.performance_data.append(performance)
        self.reward_data.append(reward)
        self.timestamps.append(len(self.timestamps))
        
        # Update line data
        self.performance_line.set_data(self.timestamps, self.performance_data)
        self.reward_line.set_data(self.timestamps, self.reward_data)
        
        # Adjust x-axis limits to show last 50 points
        if len(self.timestamps) > 50:
            self.ax1.set_xlim(len(self.timestamps) - 50, len(self.timestamps))
            self.ax2.set_xlim(len(self.timestamps) - 50, len(self.timestamps))
        else:
            self.ax1.set_xlim(0, max(50, len(self.timestamps)))
            self.ax2.set_xlim(0, max(50, len(self.timestamps)))
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
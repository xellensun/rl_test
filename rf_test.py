# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:30:33 2024

@author: yegan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces

# Step 1: Simulate weekly factor data (1000 stocks, 1040 weeks, 5 columns: momentum, value, quality, return, beta)
def simulate_factor_data(num_stocks=1000, num_weeks=1040):
    np.random.seed(42)  # For reproducibility
    # Generate random factor data
    momentum = np.random.randn(num_weeks, num_stocks)
    value = np.random.randn(num_weeks, num_stocks)
    quality = np.random.randn(num_weeks, num_stocks)
    returns = np.random.randn(num_weeks, num_stocks) * 0.02  # Simulating returns with a 2% std
    beta = np.random.randn(num_stocks) * 1  # Simulating betas, assuming an average beta of ~1

    # Combine into a single array: shape (num_weeks, num_stocks, 5) 
    factor_data = np.stack([momentum, value, quality, returns, np.tile(beta, (num_weeks, 1))], axis=2)
    return factor_data

# Simulate factor data: 1000 stocks over 1040 weeks (20 years of weekly data)
factor_data = simulate_factor_data()

# Step 2: Gym-compliant environment
class FactorTimingEnv(gym.Env):
    def __init__(self, factor_data, num_assets, training_lookback=520, evaluation_lookback=52):
        super().__init__()
        """
        Gym-compliant factor timing environment.
        - factor_data: Weekly factor data (e.g., momentum, value, quality, return, beta).
        - num_assets: Number of stocks (1000).
        - training_lookback: Lookback window for training episodes (e.g., 520 weeks for 10 years).
        - evaluation_lookback: Lookback window for policy updates (e.g., 52 weeks for 1 year).
        """
        self.factor_data = factor_data  # Historical data for momentum, value, quality, returns, beta
        self.num_assets = num_assets
        self.training_lookback = training_lookback  # Lookback for each training episode (e.g., 520 weeks)
        self.evaluation_lookback = evaluation_lookback  # Lookback for policy update (e.g., 52 weeks)
        self.current_step = 0
        self.start_step = max(training_lookback, len(factor_data) - training_lookback)  # Start with training window

        # Define action and observation spaces (gym spaces)
        self.action_space = spaces.Box(low=-0.003, high=0.003, shape=(num_assets,), dtype=np.float32)  # Weights per stock
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(training_lookback, num_assets, 4), dtype=np.float32)

    def reset(self):
        """
        Resets the environment to an initial state and returns the first observation.
        """
        self.current_step = self.start_step  # Reset to the beginning of the training window
        self.prev_action = np.zeros(self.num_assets)  # Initialize previous action (all weights = 0)
        return self.get_state()

    def get_state(self):
        """
        Returns the observation for the current step, consisting of the last 52 weeks (lookback window) of factor data.
        """
        start = max(self.current_step - self.evaluation_lookback, 0)  # Start from at least week 0
        state = self.factor_data[start:self.current_step, :, :4]  # Use 4 factors (momentum, value, quality, return)
        return state


    def step(self, action):
        """
        Apply the chosen portfolio weights (action = [w_1, w_2, ..., w_1000]).
        Enforce beta-neutrality and transaction costs.
        """
        # Enforce portfolio weight constraints: -30 bps to 30 bps
        action = np.clip(action, -0.003, 0.003)  # Between -30 bps and 30 bps
    
        # Beta data for current step
        beta_data = self.factor_data[self.current_step, :, 4]  # Shape: (1000,)
    
        # Enforce beta-neutrality: Adjust portfolio weights so that sum(beta * weights) = 0
        portfolio_beta = np.dot(beta_data, action)
        action = action - (portfolio_beta / np.dot(beta_data, beta_data)) * beta_data  # Adjust to neutralize beta exposure
    
        # Portfolio return: Weighted average of stock returns
        stock_returns = self.factor_data[self.current_step, :, 3]  # Shape: (1000,)
        portfolio_return = np.dot(action, stock_returns)
    
        # Correct transaction cost calculation
        if self.current_step > 0:
            transaction_cost = np.sum(np.abs(action - self.prev_action)) * 0.001  # 10 bps per trade
        else:
            transaction_cost = 0  # No transaction cost on the first step
    
        # Net return after transaction cost
        net_return = portfolio_return - transaction_cost
    
        # Store the current action as the previous action for the next step
        self.prev_action = action.copy()
    
        # Move to the next step
        self.current_step = self.current_step + 1
        done = self.current_step >= len(self.factor_data)
    
        next_state = self.get_state() if not done else None
        reward = net_return * 10  # Reward is the net return after transaction costs
        info = {}  # Optional debug info
        return next_state, reward, done, info

# Step 3: Define the Policy and Value RNNs
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(AttentionRNN, self).__init__()

        # LSTM layer to capture temporal dependencies
        # It takes input of size (input_size) and outputs hidden state vectors of size (hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention mechanism: Linear layer that computes attention weights
        # It reduces the hidden size down to 1 to create attention scores across time steps
        self.attention = nn.Linear(hidden_size, 1)

        # Output layer for generating portfolio weights based on the attention-weighted LSTM outputs
        self.fc = nn.Linear(hidden_size, output_size)

        # Store hidden and output sizes for reference
        self.hidden_size = hidden_size
        self.num_assets = output_size
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=0.01)
        nn.init.orthogonal_(self.rnn.weight_hh_l0, gain=0.01)
            
    def forward(self, state):
        # Get batch_size, lookback, num_assets, and num_factors from the input state
        # state shape: (batch_size, lookback, num_assets, num_factors)
        batch_size, lookback, num_assets, num_factors = state.shape

        # Step 1: Flatten the input state to be compatible with the LSTM layer
        # The LSTM expects input in the shape (batch_size, lookback, features)
        # Here, we flatten the num_assets * num_factors dimensions to create a feature vector
        state_flat = state.view(batch_size, lookback, num_assets * num_factors)

        # Step 2: Pass the flattened state through the LSTM
        # The LSTM will output hidden states for each time step, resulting in rnn_out with shape
        # (batch_size, lookback, hidden_size)
        rnn_out, _ = self.rnn(state_flat)
        rnn_out = nn.functional.relu(rnn_out)


        # Step 3: Apply the attention mechanism
        # The attention layer computes attention scores for each time step
        # Shape of attention_weights: (batch_size, lookback, 1)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)

        # Step 4: Multiply attention weights by LSTM output to focus on important time steps
        # Broadcasting is used here to multiply attention weights with rnn_out element-wise
        # This gives a weighted sum of the LSTM outputs
        weighted_rnn_out = attention_weights * rnn_out

        # Sum over the lookback dimension to collapse the temporal information into a single vector
        # Shape of weighted_rnn_out after sum: (batch_size, hidden_size)
        weighted_rnn_out = torch.sum(weighted_rnn_out, dim=1)

        # Step 5: Pass the attention-weighted LSTM output through the fully connected (FC) layer
        # The FC layer maps the hidden_size down to the number of assets (portfolio weights)
        action = self.fc(weighted_rnn_out)

        # Step 6: Clamp the output action values to keep them within a reasonable range
        # This is to avoid extreme portfolio weights
        action = torch.clamp(action, -0.003, 0.003)

        # Return the final action (portfolio weights)
        return action


# Value network to estimate V(s_t)
class ValueRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Single output value (V(s_t))
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=0.1)
        nn.init.orthogonal_(self.rnn.weight_hh_l0, gain=0.1)

    def forward(self, state):
        batch_size, lookback, num_assets, num_factors = state.shape
        state_flat = state.view(batch_size, lookback, num_assets * num_factors)

        # LSTM takes in shape (batch, lookback, features)
        rnn_out, _ = self.rnn(state_flat)
        rnn_out = nn.functional.relu(rnn_out)
        last_out = rnn_out[:, -1, :]  # Get last time step output
        value = self.fc(last_out)  # Output V(s_t)
        return value

# Step 4: Training loop with the gym-compliant environment
import time  # To track performance

def batch_trajectory(trajectory, batch_size):
    """
    Splits the trajectory into mini-batches of given size.
    """
    for i in range(0, len(trajectory), batch_size):
        yield trajectory[i:i + batch_size]
 
def track_gradients_and_losses(policy_network, value_network, policy_loss, value_loss, reward):
    # Track gradient norms for policy network
    policy_grad_norm = 0
    for param in policy_network.parameters():
        if param.grad is not None:
            policy_grad_norm += param.grad.data.norm(2).item() ** 2
    policy_gradients.append(policy_grad_norm ** 0.5)

    # Track gradient norms for value network
    value_grad_norm = 0
    for param in value_network.parameters():
        if param.grad is not None:
            value_grad_norm += param.grad.data.norm(2).item() ** 2
    value_gradients.append(value_grad_norm ** 0.5)

    # Track policy and value losses
    policy_losses.append(policy_loss.item())
    value_losses.append(value_loss.item())

    # Track rewards
    rewards_over_time.append(reward)
       
def train_rl_agent(env, policy_net, value_net, num_episodes=100, gamma=0.99, 
                   epsilon = 0.001, decay = 0.99, min_epsilon = 0.0001):
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=5e-2)
    optimizer_value = optim.Adam(value_net.parameters(), lr=5e-2)

    torch.autograd.set_detect_anomaly(False)  # Enable anomaly detection for detailed debugging

    for episode in range(num_episodes):
        print('Running Episode: ', episode + 1)
        start_episode_time = time.time()  # Track how long the episode takes

        state = env.reset()  # Reset environment at the start of each episode (520 weeks lookback)
        done = False
        total_reward = 0
        trajectory = []  # Stores the trajectory (states, actions, rewards) for the entire episode

        # Run 520 steps in each episode (10 years of data per episode)
        step_start_time = time.time()  # Track how long each step takes
        for step in range(env.training_lookback):

            # Convert state to tensor and ensure requires_grad is True
            state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True).unsqueeze(0)  # Add batch dimension

            # Step 1: Get action from policy network
            action = policy_net(state_tensor)  # Directly from the policy network
            
            # Step 2: Enforce beta-neutrality for action
            action_np = action.detach().numpy()[0]  # Convert action to numpy for the environment step
            
            if np.random.rand() < epsilon:  # Epsilon-greedy exploration
                print('Adding noise')
                action_np += np.random.normal(0, 0.001, size=action_np.shape)
            epsilon = max(epsilon * decay, min_epsilon)  # Decay epsilon over time

            beta_data = env.factor_data[env.current_step, :, 4]  # Get beta data for current step
            action_np = enforce_beta_neutrality(action_np, beta_data)  # Adjust action to be beta-neutral

            # Step 3: Step through the environment using the adjusted action
            next_state, reward, done, _ = env.step(action_np)
            
            total_reward = total_reward + reward

            # Store trajectory for training (state, action, reward)
            trajectory.append((state_tensor, action, reward))  # No detach here, keeping action in the computation graph
            
            
            state = next_state
            if done:
                break

        # Print step completion time
        print(f"Step completed in {time.time() - step_start_time:.4f} seconds")
    
        # Split the trajectory into mini-batches
        G_t = 0  
        policy_loss = torch.tensor(0.0, requires_grad=True)
        value_loss = torch.tensor(0.0, requires_grad=True)

        # Compute loss
        for t, (state_tensor, action, reward) in enumerate(reversed(trajectory)):
            G_t = reward + gamma * G_t  
            value_estimate = value_net(state_tensor)
            advantage = G_t - value_estimate.item()

            # Compute policy and value losses
            log_action_probs = torch.log_softmax(action, dim=-1) 
            policy_loss = policy_loss + (-torch.sum(log_action_probs * advantage))

            value_loss = value_loss + nn.functional.mse_loss(
                value_estimate, torch.tensor([[G_t]], dtype=torch.float32))
        
        # Backpropagation
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        
        # Normalize losses by mini_batch size for stability
        policy_loss /= len(trajectory)
        value_loss /= len(trajectory)
        
        policy_loss.backward()
        value_loss.backward()
        
        #import pdb;pdb.set_trace()
        for name, param in policy_net.named_parameters(): 
            #if param.grad is None: print(name, param)
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm()} [p]")
        
        for name, param in value_net.named_parameters():
            if param.grad is not None:
                print(f"{name} grad norm: {param.grad.norm()} [v]")

        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(),  max_norm=5.0)

        optimizer_policy.step()
        optimizer_value.step()
        

        # Track gradients and losses
        track_gradients_and_losses(policy_net, value_net, policy_loss, value_loss, reward)

        print(f"Episode {episode+1} completed, Total Reward: {total_reward}")


# Utility function for enforcing beta-neutrality
def enforce_beta_neutrality(action, beta_data):
    """
    Adjust portfolio weights to ensure the portfolio is beta-neutral.
    This ensures that the sum(beta * weights) is zero.
    """
    portfolio_beta = np.dot(beta_data, action)
    action = action - (portfolio_beta / np.dot(beta_data, beta_data)) * beta_data  # Adjust to neutralize beta exposure
    return action

import matplotlib.pyplot as plt
# Initialize lists to track gradients and losses
policy_gradients = []
value_gradients = []
value_losses = []
policy_losses = []
rewards_over_time = []

    

# Initialize the environment, policy, and value networks
input_size = 4 * 1000  # Weekly factor scores for 1000 stocks (momentum, value, quality, return)
hidden_size = 64
output_size = 1000  # Portfolio weights for 1000 stocks

policy_net = AttentionRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
value_net = ValueRNN(input_size=input_size, hidden_size=hidden_size)

env = FactorTimingEnv(factor_data=factor_data, num_assets=1000, training_lookback = 52 * 5, evaluation_lookback=52)

# Step 5: Train the agent
train_rl_agent(env, policy_net, value_net, num_episodes=10)

def plot_metrics():
    plt.figure(figsize=(15, 5))
    
    # Plot Policy Gradients
    plt.subplot(1, 3, 1)
    plt.plot(policy_gradients, label='Policy Gradients')
    plt.xlabel('Episodes')
    plt.ylabel('Gradient Norm')
    plt.title('Policy Network Gradient Norms')
    plt.legend()

    # Plot Value Gradients
    plt.subplot(1, 3, 2)
    plt.plot(value_gradients, label='Value Gradients')
    plt.xlabel('Episodes')
    plt.ylabel('Gradient Norm')
    plt.title('Value Network Gradient Norms')
    plt.legend()

    # Plot Rewards
    plt.subplot(1, 3, 3)
    plt.plot(rewards_over_time, label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot Losses (Optional)
    plt.figure(figsize=(10, 5))
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Policy and Value Losses')
    plt.legend()
    plt.show()

# After training, plot the results
plot_metrics()

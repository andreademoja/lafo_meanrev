"""
Deep Kalman Filter (DKF) Implementation
Section 3.2: Deep Kalman networks with recurrent state transition functions
Implements: Deep Kalman filter, Neural state transition, Recurrent structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict

from lafo import lafo_loss

# =============================================================================
# Neural State Transition Functions
# =============================================================================

class NeuralStateTransition(nn.Module):
    """
    Neural state transition function f(s_{t-1}, w_t).
    Implements learnable dynamics for state evolution.
    
    Section 3.2.1: Neural transition models
    """
    
    def __init__(self, state_dim: int = 4, input_dim: int = 1, hidden_dim: int = 16, 
                 nonlinear: bool = True):
        super().__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Input processing
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # State processing
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine and project to state space
        self.combine_fc = nn.Linear(hidden_dim * 2 + hidden_dim, state_dim)
        
        # If nonlinear, add activation
        self.nonlinear = nonlinear
        if nonlinear:
            self.activation = nn.Tanh()
    
    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute state transition: s_t = f(s_{t-1}, w_t)
        
        Args:
            state: Current state [B, state_dim]
            observation: Current observation [B, input_dim]
            
        Returns:
            New state [B, state_dim]
        """
        obs_processed = self.input_fc(observation)
        state_processed = self.state_fc(state)
        
        combined = torch.cat([state_processed, obs_processed], dim=-1)
        new_state = self.combine_fc(combined)
        
        if self.nonlinear:
            new_state = self.activation(new_state)
        
        return new_state


class LinearStateTransition(nn.Module):
    """
    Linear state transition for comparison.
    s_t = F s_{t-1} + G w_t
    """
    
    def __init__(self, state_dim: int = 4, input_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        
        self.F = nn.Linear(state_dim, state_dim)
        self.G = nn.Linear(input_dim, state_dim)
    
    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        new_state = self.F(state) + self.G(observation)
        return new_state


# =============================================================================
# Deep Kalman Filter Core
# =============================================================================

class DeepKalmanFilter(nn.Module):
    """
    Deep Kalman Filter implementation with recurrent structure.
    
    Section 3.2.2: Deep Kalman network architecture
    - Neural state transition functions
    - Recurrent state updates
    - Kalman gain computation
    
    Architecture:
    s_t = f(s_{t-1}, w_t)  # Neural state transition
    y_t = H s_t + v_t       # Observation model
    K_t = ...               # Kalman gain
    s_t = s_t + K_t (y_t - H s_t)  # Update
    """
    
    def __init__(self, state_dim: int = 4, input_dim: int = 1, 
                 hidden_dim: int = 16, output_dim: int = 1,
                 transition_type: str = 'neural',
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transition_type = transition_type
        
        # State transition (neural or linear)
        if transition_type == 'neural':
            self.state_transition = NeuralStateTransition(
                state_dim=state_dim,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                nonlinear=True
            )
        else:
            self.state_transition = LinearStateTransition(
                state_dim=state_dim,
                input_dim=input_dim
            )
        
        # Observation model: y_t = H s_t
        self.H = nn.Linear(state_dim, output_dim)
        
        # Kalman gain computation (learnable for DKF)
        self.k_gain_fc = nn.Sequential(
            nn.Linear(state_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Ensure gain in [0, 1]
        )
        
        # Process noise covariance (diagonal for simplicity)
        self.Q_factor = nn.Parameter(torch.ones(state_dim))
        
        # Measurement noise covariance
        self.R_factor = nn.Parameter(torch.ones(output_dim))
        
        # For recurrent structure across time
        self.recurrent_hidden = nn.Linear(output_dim, state_dim) if num_layers > 1 else None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.H.weight)
        nn.init.xavier_uniform_(self.k_gain_fc[0].weight)
        nn.init.constant_(self.H.bias, 0)
        nn.init.constant_(self.k_gain_fc[2].weight, 1.0)
        nn.init.constant_(self.k_gain_fc[2].bias, 0.5)
    
    def _linear_kalman_step(self, state: torch.Tensor, measurement: torch.Tensor,
                            H: torch.Tensor, R: torch.Tensor,
                            Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform linear Kalman update step.
        
        Args:
            state: Prior state estimate [B, state_dim]
            measurement: Observation [B, output_dim]
            H: Observation matrix [B, state_dim, output_dim]
            R: Measurement noise covariance [B, output_dim, output_dim]
            Q: Process noise covariance [B, state_dim, state_dim]
            
        Returns:
            posterior_state, posterior_covariance
        """
        B = state.size(0)
        
        # Prediction step
        prior_state = state
        prior_state_cov = torch.eye(self.state_dim).unsqueeze(0).repeat(B, 1, 1)
        
        # For simplicity, assume process noise is small
        # In full DKF, Q would be learned
        
        # Measurement covariance
        H = H.unsqueeze(1).expand(B, self.state_dim, self.output_dim)
        S = H @ prior_state_cov + R
        
        # Kalman gain
        K = prior_state_cov @ H.T @ torch.inverse(S)
        
        # Innovation
        y = measurement.unsqueeze(-1)
        innovation = y - H @ prior_state
        
        # Update
        posterior_state = prior_state + K @ innovation
        posterior_cov = (torch.eye(self.state_dim).unsqueeze(0).repeat(B, 1, 1) - 
                        K @ H) @ prior_state_cov
        
        return posterior_state, posterior_cov
    
    def forward(self, observations: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None,
                use_online_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter observations through DKF.
        
        Args:
            observations: Observations [T, B, 1] or [B, 1, T]
            initial_state: Initial state [1, state_dim] or None
            use_online_mode: If True, update state sequentially
            
        Returns:
            filtered_observations: Filtered measurements [T, B, 1] or [B, 1, T]
            posterior_states: Posterior state estimates [T, B, state_dim]
        """
        # Ensure observations are [B, 1, T]
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        elif observations.dim() == 3:
            observations = observations.permute(1, 2, 0).unsqueeze(-1)
        
        B, _, T = observations.shape
        
        # Initialize state
        if initial_state is None:
            with torch.no_grad():
                initial_state = torch.zeros(B, self.state_dim)
        
        filtered = []
        all_states = []
        
        for t in range(T):
            obs_t = observations[:, 0, t:t+1]
            
            # State transition (neural dynamics)
            state_t = self.state_transition(state_t, obs_t)
            
            # Observation model
            measurement = self.H(state_t)
            
            # For simplicity, use simplified Kalman update
            # In full implementation, use linear_kalman_step
            state_cov_diag = 1.0 + self.Q_factor.mean()
            R_val = 1.0 + self.R_factor.mean()
            
            # Simplified Kalman gain
            K_t = state_cov_diag / (state_cov_diag + R_val)
            
            # Innovation
            innovation = obs_t.squeeze(-1) - measurement.squeeze(-1)
            
            # Update state
            state_t = state_t + K_t * innovation
            
            # Store
            filtered.append(measurement)
            all_states.append(state_t)
        
        filtered = torch.stack(filtered, dim=0)  # [T, B, 1]
        all_states = torch.stack(all_states, dim=0)
        
        # Convert back to original format
        filtered = filtered.permute(1, 2, 0).squeeze(-1)  # [B, 1, T] -> [T, B, 1]
        all_states = all_states.permute(1, 2, 0)  # [T, B, state_dim]
        
        return filtered, all_states
    
    def fit(self, y: np.ndarray, K: int = 20, num_epochs: int = 100, 
            lr: float = 0.001) -> Dict:
        """
        Train the DKF using LAFO loss.
        
        Args:
            y: Observed series [T,]
            K: Smoothing window
            num_epochs: Training iterations
            lr: Learning rate
            
        Returns:
            Training history
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        
        history = {'loss': [], 'filtered': []}
        
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        T = len(y)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            filtered, states = self.forward(y_tensor)
            filtered_np = filtered.detach().cpu().numpy()
            
            # Compute LAFO loss
            loss_val = lafo_loss(y, filtered_np, K)
            loss = torch.tensor(loss_val, requires_grad=True)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                scheduler.step(loss_val)
                print(f"Epoch [{epoch+1}/{num_epochs}]  LAFO Loss: {loss.item():.6f}")
                history['loss'].append(loss.item())
                history['filtered'].append(filtered_np.copy())
        
        print("DKF training completed.")
        return history


# =============================================================================
# Recurrent Deep Kalman Filter
# =============================================================================

class RecurrentDeepKalmanFilter(nn.Module):
    """
    Recurrent Deep Kalman Filter with LSTM-style memory.
    
    Section 3.2.3: Deep recurrent structure
    Combines DKF with recurrent neural networks for better long-term dependency modeling.
    """
    
    def __init__(self, state_dim: int = 4, input_dim: int = 1,
                 hidden_dim: int = 16, output_dim: int = 1,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Neural state transition
        self.state_transition = NeuralStateTransition(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # LSTM for recurrent structure
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Observation model
        self.H = nn.Linear(hidden_dim, output_dim)
        
        # State decoder
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.H.weight)
        nn.init.xavier_uniform_(self.state_decoder[0].weight)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter with recurrent structure.
        
        Args:
            observations: [B, T, 1]
            
        Returns:
            filtered_observations, latent_states
        """
        B, T, _ = observations.shape
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(observations)
        
        # Decode to state space
        latent = torch.tanh(lstm_out)  # [B, T, hidden_dim]
        states = self.state_decoder(latent)  # [B, T, state_dim]
        
        # Observation model
        filtered = self.H(states)  # [B, T, output_dim]
        
        return filtered.squeeze(-1), states
    
    def fit(self, y: np.ndarray, K: int = 20, num_epochs: int = 100,
            lr: float = 0.001) -> Dict:
        """Train the recurrent DKF."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        T = len(y)
        
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            filtered, states = self.forward(y_tensor)
            filtered_np = filtered.detach().cpu().numpy()
            
            loss_val = lafo_loss(y, filtered_np, K)
            loss = torch.tensor(loss_val, requires_grad=True)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}]  LAFO Loss: {loss.item():.6f}")
                history['loss'].append(loss.item())
        
        print("Recurrent DKF training completed.")
        return history


# =============================================================================
# Utility Functions
# =============================================================================

def create_deep_kalman(state_dim: int = 4, input_dim: int = 1,
                        output_dim: int = 1, hidden_dim: int = 16,
                        transition_type: str = 'neural',
                        num_layers: int = 1,
                        use_recurrent: bool = False) -> nn.Module:
    """
    Factory function to create DKF model.
    
    Args:
        state_dim: State vector dimension
        input_dim: Input observation dimension
        output_dim: Output dimension
        hidden_dim: Hidden layer size
        transition_type: 'neural' or 'linear'
        num_layers: Number of recurrent layers
        use_recurrent: Use LSTM-based recurrent structure
        
    Returns:
        DeepKalmanFilter or RecurrentDeepKalmanFilter instance
    """
    if use_recurrent:
        return RecurrentDeepKalmanFilter(
            state_dim=state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers
        )
    else:
        return DeepKalmanFilter(
            state_dim=state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            transition_type=transition_type,
            num_layers=num_layers
        )

"""
Kalman Filter implementation for sensor data fusion
"""
import numpy as np
from typing import Optional, Tuple

class KalmanFilter:
    """
    Kalman Filter implementation for state estimation
    
    Attributes:
        state_dim: Dimension of the state vector
        measurement_dim: Dimension of the measurement vector
        state: Current state estimate
        covariance: Current state covariance matrix
        process_noise: Process noise covariance (Q)
        measurement_noise: Measurement noise covariance (R)
    """
    
    def __init__(self, state_dim: int, measurement_dim: int, 
                 process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize the Kalman filter
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initial state and covariance
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
        # Process noise covariance (Q)
        self.Q = np.eye(state_dim) * process_noise
        
        # Measurement noise covariance (R)
        self.R = np.eye(measurement_dim) * measurement_noise
    
    def init_state(self, initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        """
        Initialize the filter state and covariance
        
        Args:
            initial_state: Initial state vector
            initial_covariance: Initial covariance matrix
        """
        self.state = initial_state
        self.covariance = initial_covariance
    
    def predict(self, F: np.ndarray, B: Optional[np.ndarray] = None, 
                u: Optional[np.ndarray] = None) -> None:
        """
        Prediction step of the Kalman filter
        
        Args:
            F: State transition matrix
            B: Control input matrix (optional)
            u: Control input vector (optional)
        """
        # State prediction
        if B is not None and u is not None:
            self.state = F @ self.state + B @ u
        else:
            self.state = F @ self.state
        
        # Covariance prediction
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def update(self, measurement: np.ndarray, H: np.ndarray) -> None:
        """
        Update step of the Kalman filter
        
        Args:
            measurement: Measurement vector
            H: Measurement matrix
        """
        # Calculate Kalman gain
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - H @ self.state  # Measurement residual
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state estimate
        
        Returns:
            Current state vector
        """
        return self.state
    
    def get_covariance(self) -> np.ndarray:
        """
        Get the current state covariance
        
        Returns:
            Current covariance matrix
        """
        return self.covariance


class ComplementaryFilter:
    """
    Complementary Filter for fusing high and low frequency information
    
    Attributes:
        alpha: Weight for high-frequency data (0 to 1)
        state: Current state estimate
    """
    
    def __init__(self, state_dim: int, alpha: float = 0.98):
        """
        Initialize the complementary filter
        
        Args:
            state_dim: Dimension of the state vector
            alpha: Weight for high-frequency data (0 to 1)
        """
        self.state_dim = state_dim
        self.alpha = alpha
        self.state = np.zeros(state_dim)
    
    def init_state(self, initial_state: np.ndarray) -> None:
        """
        Initialize the filter state
        
        Args:
            initial_state: Initial state vector
        """
        self.state = initial_state
    
    def update(self, high_freq_data: np.ndarray, low_freq_data: np.ndarray, 
               dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the filter with new data
        
        Args:
            high_freq_data: High frequency data (e.g., gyroscope)
            low_freq_data: Low frequency data (e.g., accelerometer)
            dt: Time step
            
        Returns:
            Tuple of (filtered_state, delta)
        """
        # For gyro + accelerometer/magnetometer fusion:
        # state = alpha * (state + gyro * dt) + (1 - alpha) * accel_state
        delta = high_freq_data * dt
        high_freq_state = self.state + delta
        
        # Fuse the data
        self.state = self.alpha * high_freq_state + (1 - self.alpha) * low_freq_data
        
        return self.state, delta
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state estimate
        
        Returns:
            Current state vector
        """
        return self.state

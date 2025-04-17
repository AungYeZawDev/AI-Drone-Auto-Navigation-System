"""
Sensor simulator that generates realistic sensor readings
"""
import numpy as np
import random
from typing import Dict, Any, Tuple

class SensorSimulator:
    """
    Simulates onboard drone sensors including IMU, magnetometer, and barometer
    Generates realistic sensor readings with appropriate noise
    """
    
    def __init__(self, environment=None):
        """
        Initialize the sensor simulator with default parameters
        
        Args:
            environment: Optional environment model for realistic sensor perturbations
        """
        # Store environment model
        self.environment = environment
        
        # Sensor noise levels
        self.accel_noise_std = 0.1  # m/s^2
        self.gyro_noise_std = 0.02  # rad/s
        self.mag_noise_std = 0.1    # normalized units
        self.baro_noise_std = 0.5   # meters
        
        # Sensor biases (constant offset errors)
        self.accel_bias = np.array([0.05, -0.03, 0.02])  # m/s^2
        self.gyro_bias = np.array([0.01, -0.01, 0.01])   # rad/s
        self.mag_bias = np.array([0.05, 0.05, 0.0])      # normalized units
        self.baro_bias = 0.2  # meters
        
        # Earth's magnetic field (normalized)
        self.earth_mag_field = np.array([1.0, 0.0, 0.0])
        
        # Gravity vector (m/s^2)
        self.gravity = np.array([0.0, 0.0, 9.81])
        
        # Reference altitude and pressure
        self.reference_altitude = 0.0  # meters
        self.reference_pressure = 101325.0  # Pa (sea level standard)
        
        # Current true state (for simulation purposes)
        self.true_position = np.array([0.0, 0.0, 0.0])
        self.true_velocity = np.array([0.0, 0.0, 0.0])
        self.true_acceleration = np.array([0.0, 0.0, 0.0])
        self.true_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw in radians
        self.true_angular_velocity = np.array([0.0, 0.0, 0.0])  # rad/s
        
        # Mission pattern parameters
        self.mission_patterns = {
            'position': self._generate_position_pattern(),
            'velocity': self._generate_velocity_pattern(),
            'orientation': self._generate_orientation_pattern()
        }
    
    def _generate_position_pattern(self) -> Dict[int, np.ndarray]:
        """
        Generate a pattern of positions for the simulated flight
        
        Returns:
            Dictionary mapping timesteps to position vectors
        """
        position_pattern = {}
        
        # Define key positions at specific timesteps
        position_pattern[0] = np.array([0.0, 0.0, 0.0])
        position_pattern[5] = np.array([0.0, 0.0, 2.0])  # Takeoff to 2m
        position_pattern[10] = np.array([0.0, 0.0, 5.0])  # Climb to 5m
        position_pattern[15] = np.array([3.0, 0.0, 8.0])  # Move forward and up
        position_pattern[30] = np.array([10.0, 5.0, 10.0])  # Continue forward with some sideways
        position_pattern[45] = np.array([15.0, 12.0, 8.0])  # More movement and start descending
        position_pattern[55] = np.array([12.0, 15.0, 3.0])  # Return and descend
        position_pattern[60] = np.array([10.0, 10.0, 0.0])  # Land
        
        return position_pattern
    
    def _generate_velocity_pattern(self) -> Dict[int, np.ndarray]:
        """
        Generate a pattern of velocities for the simulated flight
        
        Returns:
            Dictionary mapping timesteps to velocity vectors
        """
        velocity_pattern = {}
        
        # Define key velocities at specific timesteps
        velocity_pattern[0] = np.array([0.0, 0.0, 0.0])
        velocity_pattern[5] = np.array([0.0, 0.0, 0.5])  # Ascend
        velocity_pattern[10] = np.array([0.5, 0.0, 0.5])  # Forward + ascend
        velocity_pattern[15] = np.array([1.0, 0.3, 0.2])  # Cruise
        velocity_pattern[30] = np.array([1.0, 1.0, 0.0])  # Turn
        velocity_pattern[45] = np.array([0.5, 0.8, -0.3])  # Descend
        velocity_pattern[55] = np.array([-0.3, 0.0, -0.5])  # Final approach
        velocity_pattern[60] = np.array([0.0, 0.0, 0.0])  # Landed
        
        return velocity_pattern
    
    def _generate_orientation_pattern(self) -> Dict[int, np.ndarray]:
        """
        Generate a pattern of orientations for the simulated flight
        
        Returns:
            Dictionary mapping timesteps to orientation vectors (roll, pitch, yaw)
        """
        orientation_pattern = {}
        
        # Define key orientations at specific timesteps (in radians)
        orientation_pattern[0] = np.array([0.0, 0.0, 0.0])
        orientation_pattern[5] = np.array([0.05, 0.05, 0.0])  # Small roll/pitch during takeoff
        orientation_pattern[10] = np.array([0.0, -0.2, 0.0])  # Pitch forward to move
        orientation_pattern[15] = np.array([0.1, -0.25, 0.3])  # Turn slightly
        orientation_pattern[30] = np.array([0.0, -0.1, 1.0])  # More significant turn
        orientation_pattern[45] = np.array([-0.1, 0.1, 2.0])  # Another direction
        orientation_pattern[55] = np.array([0.05, 0.15, 2.5])  # Approach
        orientation_pattern[60] = np.array([0.0, 0.0, 2.5])  # Landed, maintaining yaw
        
        return orientation_pattern
    
    def _interpolate_pattern(self, pattern: Dict[int, np.ndarray], timestep: int) -> np.ndarray:
        """
        Interpolate between pattern keyframes
        
        Args:
            pattern: Dictionary mapping timesteps to vector values
            timestep: Current timestep
            
        Returns:
            Interpolated vector for the current timestep
        """
        # Get all pattern timesteps
        timesteps = sorted(list(pattern.keys()))
        
        # If before first keyframe or after last, use the closest
        if timestep <= timesteps[0]:
            return pattern[timesteps[0]]
        if timestep >= timesteps[-1]:
            return pattern[timesteps[-1]]
        
        # Find surrounding keyframes
        prev_ts = max(ts for ts in timesteps if ts <= timestep)
        next_ts = min(ts for ts in timesteps if ts >= timestep)
        
        # If we're exactly on a keyframe
        if prev_ts == next_ts:
            return pattern[prev_ts]
        
        # Linear interpolation between keyframes
        prev_val = pattern[prev_ts]
        next_val = pattern[next_ts]
        
        # Calculate interpolation factor
        alpha = (timestep - prev_ts) / (next_ts - prev_ts)
        
        # Interpolate
        interpolated = prev_val + alpha * (next_val - prev_val)
        
        return interpolated
    
    def _get_true_state(self, timestep: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the true drone state for the given timestep
        
        Args:
            timestep: Current simulation timestep
            
        Returns:
            Tuple of (position, velocity, orientation)
        """
        # Get interpolated values from patterns
        position = self._interpolate_pattern(self.mission_patterns['position'], timestep)
        velocity = self._interpolate_pattern(self.mission_patterns['velocity'], timestep)
        orientation = self._interpolate_pattern(self.mission_patterns['orientation'], timestep)
        
        return position, velocity, orientation
    
    def _add_noise(self, value: np.ndarray, noise_std: float) -> np.ndarray:
        """
        Add Gaussian noise to a value
        
        Args:
            value: The base value
            noise_std: Standard deviation of the noise
            
        Returns:
            Value with added noise
        """
        noise = np.random.normal(0, noise_std, size=value.shape)
        return value + noise
    
    def _calculate_accelerometer_reading(self, position: np.ndarray, velocity: np.ndarray, 
                                        orientation: np.ndarray) -> np.ndarray:
        """
        Calculate accelerometer reading based on true state
        
        Args:
            position: True position
            velocity: True velocity
            orientation: True orientation
            
        Returns:
            Simulated accelerometer reading
        """
        # Calculate acceleration from velocity change
        # In a real simulation, we'd derive this from the flight dynamics
        # Here, we'll approximate it based on the velocity pattern
        if hasattr(self, 'prev_velocity'):
            accel = (velocity - self.prev_velocity) * 1.0  # Assuming 1Hz update rate
        else:
            accel = np.zeros(3)
        
        self.prev_velocity = velocity.copy()
        
        # Add gravity component (rotated by orientation)
        # In a proper simulation, we'd use a full rotation matrix
        roll, pitch, yaw = orientation
        
        # Simplified rotation of gravity to body frame
        gravity_component = np.array([
            self.gravity[0] * np.cos(pitch) + self.gravity[2] * np.sin(pitch),
            self.gravity[0] * np.sin(roll) * np.sin(pitch) + self.gravity[1] * np.cos(roll) 
                - self.gravity[2] * np.sin(roll) * np.cos(pitch),
            -self.gravity[0] * np.cos(roll) * np.sin(pitch) + self.gravity[1] * np.sin(roll) 
                + self.gravity[2] * np.cos(roll) * np.cos(pitch)
        ])
        
        # Combine linear acceleration and gravity
        accel_reading = accel + gravity_component
        
        # Add bias and noise
        accel_reading = accel_reading + self.accel_bias
        accel_reading = self._add_noise(accel_reading, self.accel_noise_std)
        
        return accel_reading
    
    def _calculate_gyroscope_reading(self, orientation: np.ndarray) -> np.ndarray:
        """
        Calculate gyroscope reading based on orientation change
        
        Args:
            orientation: True orientation
            
        Returns:
            Simulated gyroscope reading
        """
        # Calculate angular velocity from orientation change
        if hasattr(self, 'prev_orientation'):
            angular_velocity = (orientation - self.prev_orientation) * 1.0  # Assuming 1Hz rate
        else:
            angular_velocity = np.zeros(3)
        
        self.prev_orientation = orientation.copy()
        
        # Add random variation for more realism
        angular_velocity += np.random.normal(0, 0.05, size=3)
        
        # Add bias and noise
        gyro_reading = angular_velocity + self.gyro_bias
        gyro_reading = self._add_noise(gyro_reading, self.gyro_noise_std)
        
        return gyro_reading
    
    def _calculate_magnetometer_reading(self, orientation: np.ndarray) -> np.ndarray:
        """
        Calculate magnetometer reading based on orientation
        
        Args:
            orientation: True orientation
            
        Returns:
            Simulated magnetometer reading
        """
        # Rotate the Earth's magnetic field vector according to orientation
        roll, pitch, yaw = orientation
        
        # Simplified rotation matrix application
        # In a proper simulation, we'd use a full rotation matrix
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Simplified magnetic field reading (mainly affected by yaw)
        mag_field = np.array([
            self.earth_mag_field[0] * cos_yaw + self.earth_mag_field[1] * sin_yaw,
            -self.earth_mag_field[0] * sin_yaw + self.earth_mag_field[1] * cos_yaw,
            self.earth_mag_field[2]
        ])
        
        # Add bias and noise
        mag_reading = mag_field + self.mag_bias
        mag_reading = self._add_noise(mag_reading, self.mag_noise_std)
        
        return mag_reading
    
    def _calculate_barometer_reading(self, position: np.ndarray) -> float:
        """
        Calculate barometer altitude reading
        
        Args:
            position: True position
            
        Returns:
            Simulated barometer altitude
        """
        # Extract altitude (z component)
        true_altitude = position[2]
        
        # Add bias and noise
        baro_altitude = true_altitude + self.baro_bias
        baro_altitude = float(baro_altitude + np.random.normal(0, self.baro_noise_std))
        
        return baro_altitude
    
    def get_sensor_readings(self, timestep: int) -> Dict[str, Any]:
        """
        Get simulated sensor readings for the current timestep
        
        Args:
            timestep: Current simulation timestep
            
        Returns:
            Dictionary of sensor readings
        """
        # Get true state for this timestep
        position, velocity, orientation = self._get_true_state(timestep)
        
        # Store true state
        self.true_position = position
        self.true_velocity = velocity
        self.true_orientation = orientation
        
        # Calculate sensor readings
        accel_reading = self._calculate_accelerometer_reading(position, velocity, orientation)
        gyro_reading = self._calculate_gyroscope_reading(orientation)
        mag_reading = self._calculate_magnetometer_reading(orientation)
        baro_reading = self._calculate_barometer_reading(position)
        
        # Create sensor data dictionary
        sensor_data = {
            'accelerometer': tuple(accel_reading),
            'gyroscope': tuple(gyro_reading),
            'magnetometer': tuple(mag_reading),
            'barometer': baro_reading,
            'timestamp': timestep,
            # Include true state for debugging and visualization
            '_true_position': tuple(position),
            '_true_velocity': tuple(velocity),
            '_true_orientation': tuple(orientation)
        }
        
        return sensor_data

"""
Core drone logic and state management
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import logging

from infrastructure.sensor_simulator import SensorSimulator
from infrastructure.environment import Environment
from core.kalman_filter import KalmanFilter
from core.pid_controller import FlightController

# Set up logging
logger = logging.getLogger(__name__)

class Drone:
    """
    Represents the drone and its state
    Uses sensor data and filtering to estimate position and orientation
    """
    
    def __init__(self, sensor_simulator: SensorSimulator, environment: Optional[Environment] = None):
        """
        Initialize drone with default state and filters
        
        Args:
            sensor_simulator: Sensor simulator providing sensor readings
            environment: Optional environment model for wind and other environmental factors
        """
        # Position (x, y, z) in meters
        self.position = np.array([0.0, 0.0, 0.0])
        
        # Velocity (vx, vy, vz) in m/s
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Orientation (roll, pitch, yaw) in radians
        self.orientation = np.array([0.0, 0.0, 0.0])
        
        # Configure Kalman filters for position and orientation
        # State vector: [x, y, z, vx, vy, vz]
        self.position_filter = KalmanFilter(
            state_dim=6,
            measurement_dim=3,
            process_noise=0.01,
            measurement_noise=0.1
        )
        
        # State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.orientation_filter = KalmanFilter(
            state_dim=6,
            measurement_dim=3,
            process_noise=0.01,
            measurement_noise=0.1
        )
        
        # Reference to sensor simulator
        self.sensor_simulator = sensor_simulator
        
        # Initialize environment model or create default
        self.environment = environment if environment is not None else Environment()
        
        # Initialize filters with starting state
        self.position_filter.init_state(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Position and velocity
            np.eye(6) * 0.1  # Initial covariance
        )
        
        self.orientation_filter.init_state(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Orientation and rates
            np.eye(6) * 0.1  # Initial covariance
        )
        
        # Store sensor readings
        self.last_sensor_data: Optional[Dict[str, Any]] = None
        self.last_env_data: Optional[Dict[str, Any]] = None
        
        # Last update time (seconds)
        self.last_update_time = 0.0
        
        # Store raw and filtered data for comparison
        self.raw_position_estimate = np.array([0.0, 0.0, 0.0])
        self.raw_orientation_estimate = np.array([0.0, 0.0, 0.0])
        
        # Initialize PID flight controller
        self.flight_controller = FlightController()
        
        # Target position and orientation
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_orientation = np.array([0.0, 0.0, 0.0])
        
        # Control signals
        self.control_signals = {
            'thrust': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw_rate': 0.0
        }
        
        # Physical properties
        self.mass = 1.0  # kg
        self.moment_of_inertia = np.array([0.1, 0.1, 0.2])  # kg·m²
        
        # Keep track of external forces and moments
        self.external_forces = np.zeros(3)
        self.external_moments = np.zeros(3)
        
        logger.info("Drone initialized with Kalman filters and PID controller")
    
    def update(self, sensor_data: Dict[str, Any], timestep: int = 0) -> None:
        """
        Update drone state based on new sensor readings and environmental conditions
        
        Args:
            sensor_data: Dictionary of sensor readings
            timestep: Current simulation timestep
        """
        # Store sensor data
        self.last_sensor_data = sensor_data
        
        # Get time delta (assume 1.0 second if not specified)
        dt = 1.0  # Update rate from simulation
        
        # === Update environmental forces ===
        env_data = self.environment.update(timestep)
        self.last_env_data = env_data
        
        # Get wind force from environment
        wind_force = env_data['wind_force']
        
        # Reset external forces and moments
        self.external_forces = np.zeros(3)
        self.external_moments = np.zeros(3)
        
        # Add wind force to external forces
        self.external_forces += wind_force
        
        # Safety check for NaN values in velocity
        if np.any(np.isnan(self.velocity)):
            logger.warning("NaN detected in velocity. Resetting to zero.")
            self.velocity = np.zeros(3)
        
        # Calculate relative wind force (wind - drone_velocity)
        # Use a copy to avoid modifying the original wind force
        relative_wind = np.array(wind_force) - np.array(self.velocity)
        
        # Safety check for NaN values in relative_wind
        if np.any(np.isnan(relative_wind)):
            logger.warning("NaN detected in relative wind calculation. Using wind force only.")
            relative_wind = np.array(wind_force)
        
        # Apply maximum limits to prevent extreme values
        max_wind_component = 30.0  # m/s (reasonable maximum for simulation)
        relative_wind = np.clip(relative_wind, -max_wind_component, max_wind_component)
        
        # Calculate drag force (using simplified aerodynamic model)
        # Drag = 0.5 * rho * v^2 * Cd * A
        air_density = 1.225  # kg/m³ at sea level
        drag_coefficient = 0.5  # Simplified drag coefficient
        effective_area = 0.15  # m² (typical drone cross-section)
        
        # Altitude affects air density - with safety check
        if np.any(np.isnan(self.position)):
            logger.warning("NaN detected in position. Using default altitude.")
            altitude = 0
        else:
            altitude = self.position[2]
            
        if altitude > 0:
            # Limit altitude to reasonable values for density calculation
            safe_altitude = min(altitude, 10000)  # Cap at 10km to prevent extreme values
            air_density *= np.exp(-safe_altitude / 8000)  # Scale height ~8000m
        
        # Calculate drag magnitude with safety checks
        rel_wind_speed = np.linalg.norm(relative_wind)
        
        # Apply reasonable limits to prevent extreme values
        rel_wind_speed = min(rel_wind_speed, 50.0)  # Cap at 50 m/s (about 180 km/h)
        
        # Only calculate drag if we have meaningful wind speed
        if rel_wind_speed > 0.01:  # Small threshold to prevent division by near-zero
            # Calculate drag with safeguards against extreme values
            drag_magnitude = 0.5 * air_density * rel_wind_speed**2 * drag_coefficient * effective_area
            drag_magnitude = min(drag_magnitude, 20.0)  # Cap at 20N to prevent extreme forces
            
            drag_direction = -relative_wind / rel_wind_speed
            drag_force = drag_direction * drag_magnitude
            
            # Final safety check for NaN or inf values
            if not np.any(np.isnan(drag_force)) and not np.any(np.isinf(drag_force)):
                # Add drag to external forces
                self.external_forces += drag_force
                
                # Log significant drag for debugging
                if np.linalg.norm(drag_force) > 1.0:
                    logger.debug(f"Drag force: {drag_force}, magnitude: {drag_magnitude:.2f}N")
            else:
                logger.warning("Invalid drag force calculated. Ignoring.")
                
        # Calculate wind-induced moment (torque) with safety checks
        # Simplified model: assume offset from center of mass creates moment
        # This creates realistic rotation effects from wind
        wind_moment_arm = np.array([0.05, 0.05, 0.1])  # Offset in meters
        
        # Safety check for wind force
        safe_wind_force = np.array(wind_force)
        if np.any(np.isnan(safe_wind_force)) or np.any(np.isinf(safe_wind_force)):
            logger.warning("Invalid wind force detected. Using zero.")
            safe_wind_force = np.zeros(3)
        
        # Calculate moment with safety limits
        wind_moment = np.cross(wind_moment_arm, safe_wind_force)
        
        # Cap moment to prevent extreme values
        max_moment = 1.0  # Nm (reasonable for small drone)
        if np.linalg.norm(wind_moment) > max_moment:
            wind_moment = wind_moment / np.linalg.norm(wind_moment) * max_moment
            
        # Add to external moments after safety check
        if not np.any(np.isnan(wind_moment)) and not np.any(np.isinf(wind_moment)):
            self.external_moments += wind_moment
        
        # Extract sensor readings
        accel = np.array(sensor_data['accelerometer'])
        gyro = np.array(sensor_data['gyroscope'])
        mag = np.array(sensor_data['magnetometer'])
        baro_altitude = sensor_data['barometer']
        
        # === Raw state estimation using dead reckoning ===
        
        # Update orientation estimate from gyroscope (integrate angular velocity)
        # Simple Euler integration for roll, pitch, yaw
        self.raw_orientation_estimate += gyro * dt
        
        # Correct heading using magnetometer (simple complementary filter)
        mag_heading = np.arctan2(mag[1], mag[0])
        # Combine with 90% gyro and 10% magnetometer for yaw
        self.raw_orientation_estimate[2] = 0.9 * self.raw_orientation_estimate[2] + 0.1 * mag_heading
        
        # Acceleration in body frame -> world frame
        # (Simplified - we would normally use a rotation matrix)
        # For a full implementation, transform accel from body to world frame using quaternions
        # This is a simplified approximation
        roll, pitch, yaw = self.raw_orientation_estimate
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Simple world frame acceleration (ignoring full rotation matrix for simplicity)
        # In a real system, we'd use the full rotation matrix
        world_accel = np.array([
            accel[0] * cos_pitch * cos_yaw + accel[1] * (sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw) + accel[2] * (cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw),
            accel[0] * cos_pitch * sin_yaw + accel[1] * (sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw) + accel[2] * (cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw),
            accel[0] * (-sin_pitch) + accel[1] * sin_roll * cos_pitch + accel[2] * cos_roll * cos_pitch
        ])
        
        # Remove gravity component from z-axis
        world_accel[2] -= env_data['gravity']  # Use gravity from environment
        
        # Add external forces (converted to acceleration by dividing by mass)
        world_accel += self.external_forces / self.mass
        
        # Update velocity via integration of acceleration
        self.velocity += world_accel * dt
        
        # Update position via integration of velocity
        self.raw_position_estimate += self.velocity * dt
        
        # Override z-coordinate with barometer altitude
        self.raw_position_estimate[2] = baro_altitude
        
        # === Wind effect on orientation ===
        # Wind can cause rotation moments on the drone
        # This is a simplified model - in reality, it would depend on the drone's shape
        # and the relative angle between the wind and the drone
        
        # Calculate the relative wind vector in body frame
        # For simplicity, we'll assume the drone's body orientation
        # matches the world frame when roll and pitch are zero
        
        # Rotation matrix from world to body frame (simplified)
        R_world_to_body = np.array([
            [cos_pitch * cos_yaw, cos_pitch * sin_yaw, -sin_pitch],
            [sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw,
             sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw,
             sin_roll * cos_pitch],
            [cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw,
             cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw,
             cos_roll * cos_pitch]
        ])
        
        # Transform wind vector from world to body frame
        relative_wind_body = R_world_to_body @ wind_force
        
        # Calculate moments caused by wind (simplified model)
        # Assume the wind affects the drone's orientation based on the relative angle
        # This is a very simplified model - real aerodynamics are much more complex
        wind_moments = np.array([
            relative_wind_body[1] * 0.01,  # Effect on roll (y-component)
            -relative_wind_body[0] * 0.01,  # Effect on pitch (x-component)
            (relative_wind_body[0] - relative_wind_body[1]) * 0.005  # Effect on yaw (combined)
        ])
        
        # Add to external moments
        self.external_moments += wind_moments
        
        # Angular acceleration from external moments (τ = I·α)
        # α = τ/I (element-wise division for diagonal inertia tensor)
        angular_accel = self.external_moments / self.moment_of_inertia
        
        # Integrate angular acceleration to get effect on orientation
        # Add to the orientation estimate (simple Euler integration)
        orientation_delta = angular_accel * dt * dt * 0.5  # Second-order effect
        self.raw_orientation_estimate += orientation_delta
        
        # === Kalman filter updates ===
        
        # Position filter: state = [x, y, z, vx, vy, vz]
        # Prediction step
        # State transition matrix
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.position_filter.predict(F)
        
        # Measurement update for position - using position estimate from dead reckoning
        # For a real system, we'd have actual measurements from various sensors
        H_pos = np.zeros((3, 6))
        H_pos[0, 0] = H_pos[1, 1] = H_pos[2, 2] = 1  # Measure position directly
        
        # Safety check for NaN in raw position estimate
        if np.any(np.isnan(self.raw_position_estimate)):
            logger.warning("NaN detected in raw position estimate. Using previous filtered state.")
            # Skip the update to prevent corruption of the filter
        else:
            self.position_filter.update(self.raw_position_estimate, H_pos)
        
        # Orientation filter: state = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        # Prediction step
        F_ori = np.eye(6)
        F_ori[0, 3] = F_ori[1, 4] = F_ori[2, 5] = dt
        self.orientation_filter.predict(F_ori)
        
        # Measurement update for orientation
        H_ori = np.zeros((3, 6))
        H_ori[0, 0] = H_ori[1, 1] = H_ori[2, 2] = 1  # Measure orientation directly
        
        # Safety check for NaN in raw orientation estimate
        if np.any(np.isnan(self.raw_orientation_estimate)):
            logger.warning("NaN detected in raw orientation estimate. Using previous filtered state.")
            # Skip the update to prevent corruption of the filter
        else:
            self.orientation_filter.update(self.raw_orientation_estimate, H_ori)
        
        # Extract filtered states
        position_state = self.position_filter.state
        orientation_state = self.orientation_filter.state
        
        # Update drone state with filtered values
        self.position = position_state[:3]
        self.velocity = position_state[3:6]
        self.orientation = orientation_state[:3]
        
        # Normalize orientation angles to -π to π
        self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
        
        # === PID Controller Update ===
        
        # Prepare current state for PID controller
        current_state = {
            'position': self.get_position(),
            'orientation': self.get_orientation(),
            'velocity': self.get_velocity(),
            'external_forces': self.external_forces.copy(),  # Provide environmental forces to controller
            'external_moments': self.external_moments.copy()
        }
        
        # Compute control signals using PID controller
        self.control_signals = self.flight_controller.compute_control_signals(current_state, dt)
        
        # Log significant wind disturbances
        if np.linalg.norm(wind_force) > 2.0:
            logger.info(f"Wind disturbance: {np.linalg.norm(wind_force):.2f}N at t={timestep}")
        
        # Log control signals for debugging
        logger.debug(f"PID Control signals: {self.control_signals}")
        
        # In a real drone, these control signals would be sent to the motors
        # For the simulation, we'll use them in the FSM to influence the drone's behavior
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get the current position estimate
        
        Returns:
            Tuple of (x, y, z) coordinates
        """
        return tuple(self.position)
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """
        Get the current orientation estimate
        
        Returns:
            Tuple of (roll, pitch, yaw) angles in radians
        """
        return tuple(self.orientation)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get the current velocity estimate
        
        Returns:
            Tuple of (vx, vy, vz) velocities
        """
        return tuple(self.velocity)
    
    def set_target_position(self, x: float, y: float, z: float) -> None:
        """
        Set the target position for the drone
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            z: Target altitude (z coordinate)
        """
        self.target_position = np.array([x, y, z])
        self.flight_controller.set_target_position(x, y, z)
        logger.info(f"Set target position to ({x}, {y}, {z})")
    
    def set_target_orientation(self, yaw: float, roll: Optional[float] = None, pitch: Optional[float] = None) -> None:
        """
        Set the target orientation for the drone
        
        Args:
            yaw: Target yaw angle in radians
            roll: Optional target roll angle in radians
            pitch: Optional target pitch angle in radians
        """
        # Update target orientation array
        if roll is not None:
            self.target_orientation[0] = roll
        if pitch is not None:
            self.target_orientation[1] = pitch
        self.target_orientation[2] = yaw
        
        # Update PID controller targets
        self.flight_controller.set_target_orientation(yaw, roll, pitch)
        logger.info(f"Set target orientation to (roll={roll if roll is not None else 'unchanged'}, "
                    f"pitch={pitch if pitch is not None else 'unchanged'}, yaw={yaw})")
    
    def get_control_signals(self) -> Dict[str, float]:
        """
        Get the current control signals computed by the PID controller
        
        Returns:
            Dictionary of control signals (thrust, roll, pitch, yaw_rate)
        """
        return self.control_signals
    
    def set_thrust(self, thrust: float) -> None:
        """
        Set the drone's thrust level directly (bypassing PID controller)
        
        Args:
            thrust: Thrust level (0.0 to 1.0)
        """
        self.control_signals['thrust'] = max(0.0, min(thrust, 1.0))  # Clamp to [0, 1]
    
    def set_yaw_rate(self, yaw_rate: float) -> None:
        """
        Set the drone's target yaw rate directly (bypassing PID controller)
        
        Args:
            yaw_rate: Yaw rate in radians per second
        """
        self.control_signals['yaw_rate'] = yaw_rate
    
    def set_pitch(self, pitch: float) -> None:
        """
        Set the drone's target pitch angle directly (bypassing PID controller)
        
        Args:
            pitch: Pitch angle in radians
        """
        self.control_signals['pitch'] = pitch
    
    def set_roll(self, roll: float) -> None:
        """
        Set the drone's target roll angle directly (bypassing PID controller)
        
        Args:
            roll: Roll angle in radians
        """
        self.control_signals['roll'] = roll

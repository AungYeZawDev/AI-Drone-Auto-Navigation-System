"""
PID Controller implementation for drone flight control
"""
from typing import Dict, Tuple, List, Any, Optional
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class PIDController:
    """
    PID Controller for stable drone flight control
    
    Controls a specific aspect of the drone's flight (altitude, position, orientation)
    by continuously adjusting output based on error between setpoint and measured value.
    
    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        setpoint: Target value
        prev_error: Previous error for derivative calculation
        integral: Accumulated error for integral term
        output_limits: Optional tuple of (min, max) output values
        windup_guard: Maximum allowed value for integral term to prevent windup
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Optional[Tuple[float, float]] = None,
                 windup_guard: float = 20.0):
        """
        Initialize the PID controller with given gains
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Optional tuple of (min, max) output values
            windup_guard: Maximum allowed value for integral term to prevent windup
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.output_limits = output_limits
        self.windup_guard = windup_guard
        self.last_output = 0.0
        
        logger.debug(f"PID Controller initialized with gains P={kp}, I={ki}, D={kd}")
    
    def set_target(self, setpoint: float) -> None:
        """
        Set the target value (setpoint) for the controller
        
        Args:
            setpoint: Target value to achieve
        """
        self.setpoint = setpoint
        
    def reset(self) -> None:
        """
        Reset the controller state (integral and previous error)
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        
    def compute(self, measured_value: float, dt: float) -> float:
        """
        Compute PID output based on the current measured value
        
        Args:
            measured_value: Current measured value of the process variable
            dt: Time delta since last update in seconds
            
        Returns:
            Control output value
        """
        # Avoid division by zero
        if dt <= 0:
            return self.last_output
            
        # Calculate error
        error = self.setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral > self.windup_guard:
            self.integral = self.windup_guard
        elif self.integral < -self.windup_guard:
            self.integral = -self.windup_guard
        i_term = self.ki * self.integral
        
        # Derivative term (on measurement, not error, to avoid derivative kick)
        if dt > 0:  # Avoid division by zero
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits if defined
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        # Store error for next iteration
        self.prev_error = error
        self.last_output = output
        
        return output


class FlightController:
    """
    Flight controller using multiple PID loops for different aspects of flight
    
    Manages altitude, position, and orientation controllers to achieve stable flight.
    
    Attributes:
        altitude_pid: PID controller for altitude
        position_x_pid: PID controller for x position
        position_y_pid: PID controller for y position
        yaw_pid: PID controller for yaw angle
        roll_pid: PID controller for roll angle
        pitch_pid: PID controller for pitch angle
    """
    
    def __init__(self):
        """
        Initialize the flight controller with appropriate PID controllers
        """
        # Create PID controllers for different control aspects
        # Altitude controller - moderately aggressive, limited thrust
        self.altitude_pid = PIDController(1.0, 0.1, 0.4, output_limits=(0.0, 1.0))
        
        # Position controllers - need to output roll/pitch commands
        # Increase gains slightly for better response to disturbances
        self.position_x_pid = PIDController(0.25, 0.015, 0.07, output_limits=(-0.5, 0.5))
        self.position_y_pid = PIDController(0.25, 0.015, 0.07, output_limits=(-0.5, 0.5))
        
        # Orientation controllers - need to be responsive but not oscillatory
        self.yaw_pid = PIDController(0.7, 0.01, 0.1, output_limits=(-0.5, 0.5))
        self.roll_pid = PIDController(0.6, 0.0, 0.1, output_limits=(-0.5, 0.5))
        self.pitch_pid = PIDController(0.6, 0.0, 0.1, output_limits=(-0.5, 0.5))
        
        # Wind compensation integral terms for persistent wind
        self.wind_integral_x = 0.0
        self.wind_integral_y = 0.0
        self.wind_integral_z = 0.0
        
        logger.info("Flight controller initialized with PID loops and wind compensation")
        
    def set_target_position(self, x: float, y: float, z: float) -> None:
        """
        Set the target position for the drone
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            z: Target altitude (z coordinate)
        """
        self.position_x_pid.set_target(x)
        self.position_y_pid.set_target(y)
        self.altitude_pid.set_target(z)
        
    def set_target_orientation(self, yaw: float, 
                               roll: Optional[float] = None, 
                               pitch: Optional[float] = None) -> None:
        """
        Set the target orientation for the drone
        
        Args:
            yaw: Target yaw angle in radians
            roll: Target roll angle in radians (optional)
            pitch: Target pitch angle in radians (optional)
        """
        self.yaw_pid.set_target(yaw)
        if roll is not None:
            self.roll_pid.set_target(roll)
        if pitch is not None:
            self.pitch_pid.set_target(pitch)
            
    def compute_control_signals(self, current_state: Dict[str, Any], dt: float) -> Dict[str, float]:
        """
        Compute control signals based on current state and targets, with compensation for environmental disturbances
        
        Args:
            current_state: Dictionary containing current drone state information
            dt: Time delta since last update in seconds
            
        Returns:
            Dictionary of control signals (thrust, roll, pitch, yaw_rate)
        """
        # Extract current values from state
        position = current_state.get('position', (0, 0, 0))
        orientation = current_state.get('orientation', (0, 0, 0))
        velocity = current_state.get('velocity', (0, 0, 0))
        
        # Extract environmental forces if available
        external_forces = current_state.get('external_forces', np.zeros(3))
        external_moments = current_state.get('external_moments', np.zeros(3))
        
        current_x, current_y, current_z = position
        current_roll, current_pitch, current_yaw = orientation
        current_vx, current_vy, current_vz = velocity
        
        # Calculate environmental disturbance magnitudes
        wind_force_magnitude = np.linalg.norm(external_forces)
        wind_moment_magnitude = np.linalg.norm(external_moments)
        
        # Compute altitude control with wind compensation
        # Increase thrust when there's downward wind
        base_thrust = self.altitude_pid.compute(current_z, dt)
        
        # Enhanced wind compensation for z-axis 
        # Add extra thrust when there's downward force (negative z)
        # More aggressive compensation with higher gain
        wind_z_compensation = max(0, -external_forces[2] * 0.15)  # Increased scale factor for stronger compensation
        
        # Add additional compensation based on vertical velocity in strong wind
        if wind_force_magnitude > 3.0 and current_vz < -1.0:  
            # When falling in strong wind, add extra thrust to counteract
            vertical_vel_comp = min(0.3, -current_vz * 0.1)  # Cap at 0.3 for safety
            wind_z_compensation += vertical_vel_comp
            
        thrust = base_thrust + wind_z_compensation
        thrust = min(1.0, thrust)  # Ensure thrust doesn't exceed maximum
        
        # For position control, we need to convert world coordinates to drone's frame
        # This simplification assumes small angles, a real implementation would use rotation matrices
        
        # Adaptive gains based on wind conditions - increase aggressiveness when windy
        position_gain_factor = 1.0 + (wind_force_magnitude * 0.05)  # Increase gains up to 50% based on wind
        
        # Position control with wind compensation
        # Compute base roll/pitch targets
        base_roll_target = -self.position_y_pid.compute(current_y, dt)  # negative due to coordinate convention
        base_pitch_target = self.position_x_pid.compute(current_x, dt)
        
        # Enhanced feedforward terms to combat wind in x/y directions
        # More sophisticated model with adaptive scaling based on wind strength
        wind_comp_scale = 0.02  # Base compensation scale
        
        # Increase compensation strength for stronger winds (non-linear)
        if wind_force_magnitude > 2.0:
            # Use a progressive scaling for stronger winds
            wind_strength_factor = 1.0 + (wind_force_magnitude - 2.0) * 0.2
            wind_comp_scale *= wind_strength_factor
            
        # Apply directional compensation
        roll_wind_comp = -external_forces[1] * wind_comp_scale  # Compensate for y-axis wind
        pitch_wind_comp = external_forces[0] * wind_comp_scale  # Compensate for x-axis wind
        
        # Add integral wind compensation for persistent wind
        # This helps counter steady-state wind over time
        if wind_force_magnitude > 1.0:
            # Apply small integral factor to the horizontal wind components
            self.wind_integral_x = self.wind_integral_x * 0.95 + external_forces[0] * 0.05
            self.wind_integral_y = self.wind_integral_y * 0.95 + external_forces[1] * 0.05
            
            # Add integral component with low gain to avoid oscillations
            pitch_wind_comp += self.wind_integral_x * 0.01
            roll_wind_comp -= self.wind_integral_y * 0.01
        
        # Apply disturbance compensation
        roll_target = base_roll_target + roll_wind_comp
        pitch_target = base_pitch_target + pitch_wind_comp
        
        # Add velocity damping for better stability in wind
        roll_damping = -current_vy * 0.01
        pitch_damping = current_vx * 0.01
        
        roll_target += roll_damping
        pitch_target += pitch_damping
        
        # Compute orientation control
        roll_correction = self.roll_pid.compute(current_roll, dt)
        pitch_correction = self.pitch_pid.compute(current_pitch, dt)
        
        # Yaw control with disturbance compensation
        base_yaw_rate = self.yaw_pid.compute(current_yaw, dt)
        yaw_disturbance_comp = -external_moments[2] * 0.1  # Compensate for yaw disturbance
        yaw_rate = base_yaw_rate + yaw_disturbance_comp
        
        # Final control signals with combined corrections
        control_signals = {
            'thrust': thrust,
            'roll': roll_target + roll_correction,
            'pitch': pitch_target + pitch_correction,
            'yaw_rate': yaw_rate
        }
        
        # Log control signals and compensations for significant disturbances
        if wind_force_magnitude > 1.0:
            logger.debug(f"Wind compensation active: Force={wind_force_magnitude:.2f}N, "
                         f"Roll comp={roll_wind_comp:.3f}, Pitch comp={pitch_wind_comp:.3f}, "
                         f"Thrust comp={wind_z_compensation:.3f}")
        
        logger.debug(f"Control signals computed: {control_signals}")
        
        return control_signals
    
    def reset_controllers(self) -> None:
        """
        Reset all PID controllers
        """
        self.altitude_pid.reset()
        self.position_x_pid.reset()
        self.position_y_pid.reset()
        self.yaw_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
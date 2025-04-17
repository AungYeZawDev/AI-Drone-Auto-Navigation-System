"""
Logger for drone simulation data
"""
import logging
from typing import Dict, Any
import numpy as np

class SimulationLogger:
    """
    Logger to format and display simulation data in the console
    """
    
    def __init__(self):
        """
        Initialize the simulation logger
        """
        self.logger = logging.getLogger("drone.simulation")
    
    def log_step(self, flight_data: Dict[str, Any]) -> None:
        """
        Log data for the current simulation step
        
        Args:
            flight_data: Dictionary of flight data for the current step
        """
        # Extract data
        timestep = flight_data['timestep']
        position = flight_data['position']
        orientation = flight_data['orientation']
        velocity = flight_data['velocity']
        fsm_state = flight_data['fsm_state']
        
        # Extract sensor data
        sensor_data = flight_data['raw_sensor_data']
        accel = sensor_data['accelerometer']
        gyro = sensor_data['gyroscope']
        mag = sensor_data['magnetometer']
        baro = sensor_data['barometer']
        
        # True values (for comparison)
        true_pos = sensor_data['_true_position']
        true_ori = sensor_data['_true_orientation']
        
        # Calculate position error
        pos_error = np.sqrt(sum((np.array(position) - np.array(true_pos))**2))
        
        # Format the position and orientation for display
        position_str = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        ori_deg = tuple(np.degrees(o) for o in orientation)
        orientation_str = f"({ori_deg[0]:.2f}°, {ori_deg[1]:.2f}°, {ori_deg[2]:.2f}°)"
        velocity_str = f"({velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f})"
        
        # Format sensor readings
        accel_str = f"({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f})"
        gyro_str = f"({gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f})"
        mag_str = f"({mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f})"
        
        # Format PID controller data if available
        pid_info = ""
        if 'control_signals' in flight_data:
            control_signals = flight_data['control_signals']
            thrust = control_signals.get('thrust', 0)
            roll_ctrl = control_signals.get('roll', 0)
            pitch_ctrl = control_signals.get('pitch', 0)
            yaw_rate = control_signals.get('yaw_rate', 0)
            pid_info = f"\nPID Controls - Thrust: {thrust:.2f} | Roll: {roll_ctrl:.2f} | Pitch: {pitch_ctrl:.2f} | Yaw Rate: {yaw_rate:.2f}"
        
        # Format target position if available
        target_info = ""
        if 'target_position' in flight_data:
            target_pos = flight_data['target_position']
            target_info = f"\nTarget Position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})"
        
        # Print data to console
        print(f"Position: {position_str} | Orientation: {orientation_str}")
        print(f"Velocity: {velocity_str} | Altitude: {position[2]:.2f}m")
        print(f"Sensors - Accel: {accel_str} | Gyro: {gyro_str}")
        print(f"         Mag: {mag_str} | Baro: {baro:.2f}m")
        print(f"Position Error: {pos_error:.2f}m | State: {fsm_state}")
        
        # Print PID and target info if available
        if pid_info:
            print(pid_info)
        if target_info:
            print(target_info)
    
    def log_sensor_data(self, sensor_data: Dict[str, Any]) -> None:
        """
        Log raw sensor data
        
        Args:
            sensor_data: Dictionary of sensor readings
        """
        # Extract sensor readings
        accel = sensor_data['accelerometer']
        gyro = sensor_data['gyroscope']
        mag = sensor_data['magnetometer']
        baro = sensor_data['barometer']
        
        # Format readings for display
        accel_str = f"({accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f})"
        gyro_str = f"({gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f})"
        mag_str = f"({mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f})"
        
        # Log to console
        self.logger.debug(f"Accel: {accel_str} | Gyro: {gyro_str}")
        self.logger.debug(f"Mag: {mag_str} | Baro: {baro:.2f}m")
    
    def log_filtered_data(self, raw_data: Dict[str, Any], filtered_data: Dict[str, Any]) -> None:
        """
        Log raw and filtered data for comparison
        
        Args:
            raw_data: Dictionary of raw sensor-based estimates
            filtered_data: Dictionary of filtered estimates
        """
        # Extract raw and filtered positions
        raw_pos = raw_data['position']
        filtered_pos = filtered_data['position']
        
        # Format for display
        raw_pos_str = f"({raw_pos[0]:.2f}, {raw_pos[1]:.2f}, {raw_pos[2]:.2f})"
        filtered_pos_str = f"({filtered_pos[0]:.2f}, {filtered_pos[1]:.2f}, {filtered_pos[2]:.2f})"
        
        # Log to console
        self.logger.debug(f"Raw position: {raw_pos_str}")
        self.logger.debug(f"Filtered position: {filtered_pos_str}")

"""
Visualize drone flight data using matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from mpl_toolkits.mplot3d import Axes3D

class FlightPlotter:
    """
    Creates visualizations of the drone's flight path and sensor data
    """
    
    def __init__(self):
        """
        Initialize the flight plotter
        """
        self.figures = []
    
    def plot_flight_path(self, flight_data: List[Dict[str, Any]]) -> None:
        """
        Create a 3D plot of the drone flight path
        
        Args:
            flight_data: List of flight data records
        """
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract position data
        positions = np.array([record['position'] for record in flight_data])
        true_positions = np.array([record['raw_sensor_data']['_true_position'] for record in flight_data])
        
        # Plot estimated and true trajectories
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Estimated Path')
        ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], 'g--', linewidth=2, label='True Path')
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, marker='^', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, marker='v', label='End')
        
        # Add drone position markers at regular intervals
        interval = max(1, len(flight_data) // 10)  # Show up to 10 markers
        for i in range(0, len(flight_data), interval):
            if i > 0 and i < len(flight_data) - 1:  # Skip start and end
                ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], color='blue', s=50, alpha=0.5)
        
        # Add text annotations for time at key points
        for i in range(0, len(flight_data), interval):
            if i > 0:  # Skip start
                ax.text(positions[i, 0], positions[i, 1], positions[i, 2], f't={i}s', fontsize=8)
        
        # Plot orientation vectors at regular intervals (simplified)
        # This shows the drone's heading
        for i in range(0, len(flight_data), interval):
            pos = positions[i]
            yaw = flight_data[i]['orientation'][2]  # Get yaw angle
            # Create a direction vector based on yaw
            dir_vec = np.array([np.cos(yaw), np.sin(yaw), 0]) * 0.5
            ax.quiver(pos[0], pos[1], pos[2], dir_vec[0], dir_vec[1], dir_vec[2], 
                     color='red', length=1.0, normalize=True, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Drone Flight Path (Estimated vs True)')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio to prevent distortion
        ax.set_box_aspect([1, 1, 0.5])  # Adjust for a better view
        
        # Add grid
        ax.grid(True)
        
        # Store figure
        self.figures.append(fig)
    
    def plot_sensor_data(self, flight_data: List[Dict[str, Any]]) -> None:
        """
        Create plots of sensor data over time
        
        Args:
            flight_data: List of flight data records
        """
        # Create figure for position data
        fig_pos = plt.figure(figsize=(12, 8))
        fig_pos.suptitle('Position and Altitude Over Time')
        
        # Time axis
        timesteps = [record['timestep'] for record in flight_data]
        
        # Position subplots
        ax1 = fig_pos.add_subplot(311)
        ax1.set_ylabel('X Position (m)')
        ax1.grid(True)
        
        ax2 = fig_pos.add_subplot(312)
        ax2.set_ylabel('Y Position (m)')
        ax2.grid(True)
        
        ax3 = fig_pos.add_subplot(313)
        ax3.set_ylabel('Z Position (m)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        
        # Extract position data
        x_pos = [record['position'][0] for record in flight_data]
        y_pos = [record['position'][1] for record in flight_data]
        z_pos = [record['position'][2] for record in flight_data]
        
        # Extract true position data
        true_x = [record['raw_sensor_data']['_true_position'][0] for record in flight_data]
        true_y = [record['raw_sensor_data']['_true_position'][1] for record in flight_data]
        true_z = [record['raw_sensor_data']['_true_position'][2] for record in flight_data]
        
        # Plot position data
        ax1.plot(timesteps, x_pos, 'b-', label='Estimated')
        ax1.plot(timesteps, true_x, 'g--', label='True')
        ax1.legend()
        
        ax2.plot(timesteps, y_pos, 'b-', label='Estimated')
        ax2.plot(timesteps, true_y, 'g--', label='True')
        ax2.legend()
        
        ax3.plot(timesteps, z_pos, 'b-', label='Estimated')
        ax3.plot(timesteps, true_z, 'g--', label='True')
        ax3.legend()
        
        # Adjust spacing
        fig_pos.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create figure for orientation data
        fig_ori = plt.figure(figsize=(12, 8))
        fig_ori.suptitle('Orientation Over Time')
        
        # Orientation subplots
        ax4 = fig_ori.add_subplot(311)
        ax4.set_ylabel('Roll (degrees)')
        ax4.grid(True)
        
        ax5 = fig_ori.add_subplot(312)
        ax5.set_ylabel('Pitch (degrees)')
        ax5.grid(True)
        
        ax6 = fig_ori.add_subplot(313)
        ax6.set_ylabel('Yaw (degrees)')
        ax6.set_xlabel('Time (s)')
        ax6.grid(True)
        
        # Extract orientation data and convert to degrees
        roll = [np.degrees(record['orientation'][0]) for record in flight_data]
        pitch = [np.degrees(record['orientation'][1]) for record in flight_data]
        yaw = [np.degrees(record['orientation'][2]) for record in flight_data]
        
        # Extract true orientation data
        true_roll = [np.degrees(record['raw_sensor_data']['_true_orientation'][0]) for record in flight_data]
        true_pitch = [np.degrees(record['raw_sensor_data']['_true_orientation'][1]) for record in flight_data]
        true_yaw = [np.degrees(record['raw_sensor_data']['_true_orientation'][2]) for record in flight_data]
        
        # Plot orientation data
        ax4.plot(timesteps, roll, 'b-', label='Estimated')
        ax4.plot(timesteps, true_roll, 'g--', label='True')
        ax4.legend()
        
        ax5.plot(timesteps, pitch, 'b-', label='Estimated')
        ax5.plot(timesteps, true_pitch, 'g--', label='True')
        ax5.legend()
        
        ax6.plot(timesteps, yaw, 'b-', label='Estimated')
        ax6.plot(timesteps, true_yaw, 'g--', label='True')
        ax6.legend()
        
        # Adjust spacing
        fig_ori.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Create figure for sensor data
        fig_sensor = plt.figure(figsize=(12, 10))
        fig_sensor.suptitle('Sensor Readings Over Time')
        
        # Accelerometer subplot
        ax7 = fig_sensor.add_subplot(411)
        ax7.set_ylabel('Acceleration (m/s²)')
        ax7.grid(True)
        
        # Gyroscope subplot
        ax8 = fig_sensor.add_subplot(412)
        ax8.set_ylabel('Angular Velocity (rad/s)')
        ax8.grid(True)
        
        # Magnetometer subplot
        ax9 = fig_sensor.add_subplot(413)
        ax9.set_ylabel('Magnetic Field')
        ax9.grid(True)
        
        # Barometer subplot
        ax10 = fig_sensor.add_subplot(414)
        ax10.set_ylabel('Altitude (m)')
        ax10.set_xlabel('Time (s)')
        ax10.grid(True)
        
        # Extract sensor data
        accel_x = [record['raw_sensor_data']['accelerometer'][0] for record in flight_data]
        accel_y = [record['raw_sensor_data']['accelerometer'][1] for record in flight_data]
        accel_z = [record['raw_sensor_data']['accelerometer'][2] for record in flight_data]
        
        gyro_x = [record['raw_sensor_data']['gyroscope'][0] for record in flight_data]
        gyro_y = [record['raw_sensor_data']['gyroscope'][1] for record in flight_data]
        gyro_z = [record['raw_sensor_data']['gyroscope'][2] for record in flight_data]
        
        mag_x = [record['raw_sensor_data']['magnetometer'][0] for record in flight_data]
        mag_y = [record['raw_sensor_data']['magnetometer'][1] for record in flight_data]
        mag_z = [record['raw_sensor_data']['magnetometer'][2] for record in flight_data]
        
        baro = [record['raw_sensor_data']['barometer'] for record in flight_data]
        
        # Plot sensor data
        ax7.plot(timesteps, accel_x, 'r-', label='X')
        ax7.plot(timesteps, accel_y, 'g-', label='Y')
        ax7.plot(timesteps, accel_z, 'b-', label='Z')
        ax7.legend()
        
        ax8.plot(timesteps, gyro_x, 'r-', label='Roll Rate')
        ax8.plot(timesteps, gyro_y, 'g-', label='Pitch Rate')
        ax8.plot(timesteps, gyro_z, 'b-', label='Yaw Rate')
        ax8.legend()
        
        ax9.plot(timesteps, mag_x, 'r-', label='X')
        ax9.plot(timesteps, mag_y, 'g-', label='Y')
        ax9.plot(timesteps, mag_z, 'b-', label='Z')
        ax9.legend()
        
        ax10.plot(timesteps, baro, 'b-', label='Barometer')
        ax10.plot(timesteps, true_z, 'g--', label='True Altitude')
        ax10.legend()
        
        # Adjust spacing
        fig_sensor.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Store figures
        self.figures.extend([fig_pos, fig_ori, fig_sensor])
        
        # Create error analysis figure
        fig_error = plt.figure(figsize=(12, 8))
        fig_error.suptitle('Position Estimation Error')
        
        # Position error
        ax_err = fig_error.add_subplot(111)
        ax_err.set_ylabel('Error (m)')
        ax_err.set_xlabel('Time (s)')
        ax_err.grid(True)
        
        # Calculate error
        x_error = [abs(est - true) for est, true in zip(x_pos, true_x)]
        y_error = [abs(est - true) for est, true in zip(y_pos, true_y)]
        z_error = [abs(est - true) for est, true in zip(z_pos, true_z)]
        total_error = [np.sqrt(ex**2 + ey**2 + ez**2) for ex, ey, ez in zip(x_error, y_error, z_error)]
        
        # Plot errors
        ax_err.plot(timesteps, x_error, 'r-', label='X Error')
        ax_err.plot(timesteps, y_error, 'g-', label='Y Error')
        ax_err.plot(timesteps, z_error, 'b-', label='Z Error')
        ax_err.plot(timesteps, total_error, 'k-', label='Total Error')
        ax_err.legend()
        
        # Store figure
        self.figures.append(fig_error)
    
    def plot_kalman_comparison(self, flight_data: List[Dict[str, Any]]) -> None:
        """
        Create plots comparing raw and filtered data
        
        Args:
            flight_data: List of flight data records
        """
        # Implementation would compare raw sensor-based estimates to filtered estimates
        # This is a placeholder for a more complete implementation
        pass
    
    def plot_pid_controller(self, flight_data: List[Dict[str, Any]]) -> None:
        """
        Create plots showing PID controller performance
        
        Args:
            flight_data: List of flight data records
        """
        # Create figure for PID control signals
        fig_pid = plt.figure(figsize=(12, 10))
        fig_pid.suptitle('PID Controller Performance')
        
        # Time axis
        timesteps = list(range(len(flight_data)))
        
        # Extract control signals if available
        if 'control_signals' in flight_data[0]:
            # Control signals subplot
            ax1 = fig_pid.add_subplot(411)
            ax1.set_ylabel('Thrust')
            ax1.set_title('Control Signals')
            ax1.grid(True)
            
            ax2 = fig_pid.add_subplot(412)
            ax2.set_ylabel('Roll (rad)')
            ax2.grid(True)
            
            ax3 = fig_pid.add_subplot(413)
            ax3.set_ylabel('Pitch (rad)')
            ax3.grid(True)
            
            ax4 = fig_pid.add_subplot(414)
            ax4.set_ylabel('Yaw Rate (rad/s)')
            ax4.set_xlabel('Time (s)')
            ax4.grid(True)
            
            # Extract control signal data
            thrust = [record.get('control_signals', {}).get('thrust', 0) for record in flight_data]
            roll = [record.get('control_signals', {}).get('roll', 0) for record in flight_data]
            pitch = [record.get('control_signals', {}).get('pitch', 0) for record in flight_data]
            yaw_rate = [record.get('control_signals', {}).get('yaw_rate', 0) for record in flight_data]
            
            # Plot control signals
            ax1.plot(timesteps, thrust, 'r-')
            ax2.plot(timesteps, roll, 'g-')
            ax3.plot(timesteps, pitch, 'b-')
            ax4.plot(timesteps, yaw_rate, 'k-')
            
            # Create figure for setpoint tracking
            fig_tracking = plt.figure(figsize=(12, 10))
            fig_tracking.suptitle('PID Setpoint Tracking')
            
            # Extract target position and orientation if available
            if 'target_position' in flight_data[0] and 'target_orientation' in flight_data[0]:
                # Position tracking subplot
                ax5 = fig_tracking.add_subplot(311)
                ax5.set_ylabel('X Position (m)')
                ax5.set_title('Position Tracking')
                ax5.grid(True)
                
                ax6 = fig_tracking.add_subplot(312)
                ax6.set_ylabel('Y Position (m)')
                ax6.grid(True)
                
                ax7 = fig_tracking.add_subplot(313)
                ax7.set_ylabel('Z Position (m)')
                ax7.set_xlabel('Time (s)')
                ax7.grid(True)
                
                # Extract actual position data
                x_pos = [record['position'][0] for record in flight_data]
                y_pos = [record['position'][1] for record in flight_data]
                z_pos = [record['position'][2] for record in flight_data]
                
                # Extract target position data
                target_x = [record.get('target_position', [0, 0, 0])[0] for record in flight_data]
                target_y = [record.get('target_position', [0, 0, 0])[1] for record in flight_data]
                target_z = [record.get('target_position', [0, 0, 0])[2] for record in flight_data]
                
                # Plot position tracking
                ax5.plot(timesteps, x_pos, 'b-', label='Actual')
                ax5.plot(timesteps, target_x, 'r--', label='Target')
                ax5.legend()
                
                ax6.plot(timesteps, y_pos, 'b-', label='Actual')
                ax6.plot(timesteps, target_y, 'r--', label='Target')
                ax6.legend()
                
                ax7.plot(timesteps, z_pos, 'b-', label='Actual')
                ax7.plot(timesteps, target_z, 'r--', label='Target')
                ax7.legend()
                
                # Store figure
                self.figures.append(fig_tracking)
            
            # Adjust spacing
            fig_pid.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Store figure
            self.figures.append(fig_pid)
    
    def plot_environment(self, flight_data: List[Dict[str, Any]]) -> None:
        """
        Create plots showing environmental data including wind forces and effects
        
        Args:
            flight_data: List of flight data records
        """
        # Create figure for environmental data
        fig_env = plt.figure(figsize=(12, 12))
        fig_env.suptitle('Environmental Forces and Effects')
        
        # Time axis
        timesteps = list(range(len(flight_data)))
        
        # Check if environmental data is available
        if 'env_data' in flight_data[0]:
            # Wind force magnitude subplot
            ax1 = fig_env.add_subplot(411)
            ax1.set_title('Wind Force Magnitude')
            ax1.set_ylabel('Force (N)')
            ax1.grid(True)
            
            # Wind force components subplot
            ax2 = fig_env.add_subplot(412)
            ax2.set_title('Wind Force Components')
            ax2.set_ylabel('Force (N)')
            ax2.grid(True)
            
            # Drag force subplot
            ax3 = fig_env.add_subplot(413)
            ax3.set_title('Drone External Forces')
            ax3.set_ylabel('Force (N)')
            ax3.grid(True)
            
            # Environmental parameters subplot
            ax4 = fig_env.add_subplot(414)
            ax4.set_title('Environmental Parameters')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Value')
            ax4.grid(True)
            
            # Extract environmental data
            wind_force_magnitude = []
            wind_force_x = []
            wind_force_y = []
            wind_force_z = []
            
            external_force_magnitude = []
            external_force_x = []
            external_force_y = []
            external_force_z = []
            
            temperature = []
            pressure = []
            
            # Gather data from flight records
            for record in flight_data:
                env_data = record.get('env_data', {})
                
                # Wind forces
                wind_force = env_data.get('wind_force', np.zeros(3))
                if isinstance(wind_force, list) or isinstance(wind_force, np.ndarray):
                    if len(wind_force) == 3:
                        mag = np.linalg.norm(wind_force)
                        wx, wy, wz = wind_force
                    else:
                        # Handle wrong length
                        mag = 0
                        wx, wy, wz = 0, 0, 0
                else:
                    # Handle non-array or invalid data
                    mag = 0
                    wx, wy, wz = 0, 0, 0
                    
                wind_force_magnitude.append(mag)
                wind_force_x.append(wx)
                wind_force_y.append(wy)
                wind_force_z.append(wz)
                
                # External forces applied to drone
                ext_forces = record.get('external_forces', np.zeros(3))
                if isinstance(ext_forces, list) or isinstance(ext_forces, np.ndarray):
                    if len(ext_forces) == 3:
                        ext_mag = np.linalg.norm(ext_forces)
                        ex, ey, ez = ext_forces
                    else:
                        # Handle wrong length
                        ext_mag = 0
                        ex, ey, ez = 0, 0, 0
                else:
                    # Handle non-array or invalid data
                    ext_mag = 0
                    ex, ey, ez = 0, 0, 0
                    
                external_force_magnitude.append(ext_mag)
                external_force_x.append(ex)
                external_force_y.append(ey)
                external_force_z.append(ez)
                
                # Environmental parameters
                temperature.append(env_data.get('temperature', 20.0))
                pressure.append(env_data.get('pressure', 101325.0) / 1000.0)  # kPa for better scaling
            
            # Plot wind force magnitude
            ax1.plot(timesteps, wind_force_magnitude, 'b-', label='Wind Magnitude')
            ax1.legend()
            
            # Plot wind force components
            ax2.plot(timesteps, wind_force_x, 'r-', label='Wind Force X')
            ax2.plot(timesteps, wind_force_y, 'g-', label='Wind Force Y')
            ax2.plot(timesteps, wind_force_z, 'b-', label='Wind Force Z')
            ax2.legend()
            
            # Plot external forces on drone
            ax3.plot(timesteps, external_force_magnitude, 'k-', label='Total External Force')
            ax3.plot(timesteps, external_force_x, 'r--', label='External Force X')
            ax3.plot(timesteps, external_force_y, 'g--', label='External Force Y')
            ax3.plot(timesteps, external_force_z, 'b--', label='External Force Z')
            ax3.legend()
            
            # Plot environmental parameters
            ax4.plot(timesteps, temperature, 'r-', label='Temperature (°C)')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(timesteps, pressure, 'b-', label='Pressure (kPa)')
            ax4_twin.set_ylabel('Pressure (kPa)')
            
            # Combine legends from both y-axes
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            self.figures.append(fig_env)

    def save_plots(self, dir_path: str) -> None:
        """
        Save all generated plots to files
        
        Args:
            dir_path: Directory path to save plots to
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save each figure to a file
        for i, fig in enumerate(self.figures):
            filename = os.path.join(dir_path, f"drone_plot_{i}.png")
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {filename}")
    
    def show(self) -> None:
        """
        Show all generated plots
        """
        plt.show()

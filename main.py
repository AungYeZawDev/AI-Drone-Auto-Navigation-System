#!/usr/bin/env python3
"""
Autonomous Drone Navigation Simulation
Main module that orchestrates the drone simulation
"""
import os
import time
from typing import Dict, List, Any, Optional
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import simulation components
from core.drone import Drone
from core.fsm import FlightStateMachine, FlightState
from infrastructure.sensor_simulator import SensorSimulator
from infrastructure.environment import Environment
from presentation.logger import SimulationLogger
from presentation.plotter import FlightPlotter

# For web interface integration
from app import app

# Global variables for simulation state
simulation_thread = None
simulation_running = False
current_simulation_data = {}
flight_data_history = []

def run_simulation(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Run the drone simulation with the given configuration
    
    Args:
        config: Configuration dictionary for the simulation
        
    Returns:
        List of flight data records
    """
    global simulation_running, current_simulation_data
    
    # Initialize empty config if None provided
    if config is None:
        config = {}
    
    # Simulation parameters
    simulation_duration = config.get('duration', 60)  # seconds
    update_rate = config.get('update_rate', 1)  # Hz
    
    # Environment configuration
    env_config = config.get('environment', {})
    wind_scenario = env_config.get('wind_scenario', 'moderate')
    enable_wind = env_config.get('enable_wind', True)
    
    # Initialize components
    logger.info("Initializing simulation components...")
    
    # Create environment
    environment = Environment(wind_scenario=wind_scenario, enable_wind=enable_wind)
    
    # Create a sensor simulator
    sensor_simulator = SensorSimulator(environment=environment)
    
    # Create the drone with the sensor simulator and environment
    drone = Drone(sensor_simulator, environment=environment)
    
    # Apply PID configuration if provided
    if config and 'drone' in config and 'pid_gains' in config['drone']:
        pid_config = config['drone']['pid_gains']
        # In a full implementation, this would update the PID controller gains
        logger.info(f"Applying PID configuration: {pid_config}")
    
    # Create the flight state machine
    fsm = FlightStateMachine(drone)
    
    # Create a logger
    sim_logger = SimulationLogger()
    
    # Storage for flight data
    flight_data: List[Dict[str, Any]] = []
    
    # Run the simulation
    logger.info("Starting drone simulation...")
    print("\n=== DRONE NAVIGATION SIMULATION ===")
    print(f"Duration: {simulation_duration} seconds | Update rate: {update_rate}Hz")
    print(f"Wind scenario: {wind_scenario} | Wind enabled: {enable_wind}\n")
    
    # Set initial state
    fsm.set_state(FlightState.IDLE)
    
    # Initialize matplotlib for non-interactive plotting
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    simulation_running = True
    
    for timestep in range(simulation_duration):
        if not simulation_running:
            logger.info("Simulation stopped early.")
            break
            
        print(f"\n--- Timestep: {timestep} ---")
        
        # Get sensor readings
        sensor_data = sensor_simulator.get_sensor_readings(timestep)
        
        # Update drone state based on sensor readings
        drone.update(sensor_data, timestep=timestep)
        
        # Get environmental forces
        env_state = environment.update(timestep, drone.get_position(), drone.get_velocity())
        
        # Update FSM and get next command
        fsm.update(timestep)
        
        # Calculate wind force magnitude
        wind_force = 0.0
        if 'wind_force' in env_state:
            import numpy as np
            wind_force = np.linalg.norm(env_state['wind_force'])
        
        # Log the current state with additional PID controller data
        flight_record = {
            'timestep': timestep,
            'raw_sensor_data': sensor_data,
            'position': drone.get_position(),
            'orientation': drone.get_orientation(),
            'velocity': drone.get_velocity(),
            'fsm_state': fsm.current_state.name,
            'target_position': drone.target_position.tolist(),
            'target_orientation': drone.target_orientation.tolist(),
            'control_signals': drone.get_control_signals(),
            'environment': env_state,
            'wind_force': wind_force
        }
        flight_data.append(flight_record)
        
        # Update current state for web interface
        current_simulation_data = {
            'timestep': timestep,
            'position': drone.get_position(),
            'orientation': drone.get_orientation(),
            'velocity': drone.get_velocity(),
            'fsm_state': fsm.current_state.name,
            'wind_force': wind_force
        }
        
        # Log the current status
        sim_logger.log_step(flight_record)
        
        # Create plots every 10 timesteps
        if timestep % 10 == 0 or timestep == simulation_duration - 1:
            update_plots(flight_data)
        
        # Sleep to maintain update rate (disabled for headless execution)
        # time.sleep(1/update_rate)
    
    # Simulation complete
    logger.info("Simulation complete.")
    print("\n=== SIMULATION COMPLETE ===")
    
    # Final plot update
    update_plots(flight_data)
    
    # Mark simulation as complete
    simulation_running = False
    
    return flight_data


def update_plots(flight_data: List[Dict[str, Any]]) -> None:
    """
    Update visualization plots with current flight data
    
    Args:
        flight_data: List of flight data records
    """
    try:
        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        # Create plots
        plotter = FlightPlotter()
        plotter.plot_flight_path(flight_data)
        plotter.plot_sensor_data(flight_data)
        plotter.plot_kalman_comparison(flight_data)
        plotter.plot_pid_controller(flight_data)
        plotter.plot_environment(flight_data)
        
        # Save figures to directory
        plotter.save_plots('plots')
        
        logger.debug(f"Updated plots with {len(flight_data)} data points")
    except Exception as e:
        logger.error(f"Error updating plots: {str(e)}")


def start_simulation_thread(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Start a new simulation in a separate thread
    
    Args:
        config: Configuration dictionary for the simulation
    """
    global simulation_thread, simulation_running, flight_data_history
    
    if simulation_running:
        logger.warning("Simulation already running, ignoring start request")
        return
    
    # Clear previous data
    current_simulation_data.clear()
    
    # Start simulation in a new thread
    simulation_thread = threading.Thread(target=lambda: flight_data_history.extend(run_simulation(config)))
    simulation_thread.daemon = True
    simulation_thread.start()
    
    logger.info("Simulation thread started")


def stop_simulation() -> None:
    """
    Stop the currently running simulation
    """
    global simulation_running
    
    if not simulation_running:
        logger.warning("No simulation running, ignoring stop request")
        return
    
    logger.info("Stopping simulation...")
    simulation_running = False


def get_simulation_status() -> Dict[str, Any]:
    """
    Get the current status of the simulation
    
    Returns:
        Dictionary with simulation status information
    """
    return {
        'running': simulation_running,
        'current_state': current_simulation_data
    }


def main() -> None:
    """
    Main function to run the drone simulation directly
    """
    # Run the simulation with default settings
    run_simulation()


if __name__ == "__main__":
    # If run as a script, execute the simulation directly
    main()

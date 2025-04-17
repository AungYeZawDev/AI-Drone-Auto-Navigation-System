"""
Finite State Machine for drone flight control
"""
from enum import Enum, auto
from typing import List, Dict, Any
import logging
import numpy as np

from core.drone import Drone

logger = logging.getLogger(__name__)

class FlightState(Enum):
    """
    Possible states for the drone's flight state machine
    """
    IDLE = auto()
    TAKEOFF = auto()
    HOVER = auto()
    ASCEND = auto()
    DESCEND = auto()
    FORWARD = auto()
    ROTATE = auto()
    LAND = auto()

class FlightStateMachine:
    """
    Finite State Machine to control drone flight behavior
    
    Controls transitions between different flight states and
    generates appropriate commands for the drone.
    """
    
    def __init__(self, drone: Drone, initial_state: FlightState = FlightState.IDLE):
        """
        Initialize the FSM with a drone and initial state
        
        Args:
            drone: The drone to control
            initial_state: Initial flight state
        """
        self.drone = drone
        self.current_state = initial_state
        self.previous_state = initial_state
        
        # Track time spent in each state
        self.state_entry_time = 0
        
        # Define the planned mission as a sequence of states and durations
        self.mission_plan: List[Dict[str, Any]] = [
            {'state': FlightState.TAKEOFF, 'duration': 5},
            {'state': FlightState.HOVER, 'duration': 5},
            {'state': FlightState.ASCEND, 'duration': 10},
            {'state': FlightState.FORWARD, 'duration': 15},
            {'state': FlightState.ROTATE, 'duration': 10},
            {'state': FlightState.DESCEND, 'duration': 10},
            {'state': FlightState.LAND, 'duration': 5}
        ]
        
        # Current mission step
        self.mission_index = 0
        
        logger.info(f"Flight State Machine initialized in {self.current_state.name} state")
    
    def set_state(self, new_state: FlightState) -> None:
        """
        Transition to a new state
        
        Args:
            new_state: The new state to transition to
        """
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            logger.info(f"State transition: {self.previous_state.name} -> {self.current_state.name}")
            
            # Reset state entry time
            self.state_entry_time = 0
    
    def update(self, timestep: int) -> None:
        """
        Update the FSM based on current conditions and mission plan
        
        Args:
            timestep: Current simulation timestep
        """
        # Increment time in current state
        self.state_entry_time += 1
        
        # Drone position for decision making
        position = self.drone.get_position()
        
        # Follow the mission plan
        if self.mission_index < len(self.mission_plan):
            current_step = self.mission_plan[self.mission_index]
            
            # Check if we need to transition to the next step
            if self.current_state != current_step['state']:
                self.set_state(current_step['state'])
            
            # Check if we've completed the current step
            if self.state_entry_time >= current_step['duration']:
                self.mission_index += 1
                if self.mission_index < len(self.mission_plan):
                    next_step = self.mission_plan[self.mission_index]
                    self.set_state(next_step['state'])
        
        # Execute behavior based on current state
        self._execute_state_behavior()
        
        # Print current state information
        print(f"FSM State: {self.current_state.name}, Time in state: {self.state_entry_time}s")
    
    def _execute_state_behavior(self) -> None:
        """
        Execute the behavior for the current state using PID control
        """
        # Current drone position and orientation
        position = self.drone.get_position()
        orientation = self.drone.get_orientation()
        velocity = self.drone.get_velocity()
        
        # Get the starting position (can be used as a reference)
        starting_position = (0, 0, 0)
        
        # Execute behavior based on current state
        if self.current_state == FlightState.IDLE:
            # Do nothing, wait for takeoff command
            # Reset all PID controllers to ensure clean start
            self.drone.set_target_position(0, 0, 0)
            self.drone.set_target_orientation(0, 0, 0)
            
        elif self.current_state == FlightState.TAKEOFF:
            # Use PID to control takeoff to target altitude of 2m
            # Set moderate ascent speed with level orientation
            self.drone.set_target_position(position[0], position[1], 2.0)
            self.drone.set_target_orientation(orientation[2], 0.0, 0.0)  # Maintain current heading
            
        elif self.current_state == FlightState.HOVER:
            # Use PID to maintain current position with stable hover
            # Hold the current position with level orientation
            self.drone.set_target_position(position[0], position[1], position[2])
            self.drone.set_target_orientation(orientation[2], 0.0, 0.0)
            
        elif self.current_state == FlightState.ASCEND:
            # Use PID to control ascent to 10m
            # We want to move mostly vertically during this state
            target_altitude = 10.0
            self.drone.set_target_position(position[0], position[1], target_altitude)
            self.drone.set_target_orientation(orientation[2], 0.0, 0.0)
            
        elif self.current_state == FlightState.FORWARD:
            # Use PID to control forward flight with stable altitude
            # Create a target 50m ahead of the current position in the drone's current heading
            # Using the drone's yaw to determine forward direction
            yaw = orientation[2]
            distance = 50.0
            target_x = position[0] + distance * np.cos(yaw)
            target_y = position[1] + distance * np.sin(yaw)
            
            # Set a target that's ahead of the drone, maintaining current altitude
            self.drone.set_target_position(target_x, target_y, position[2])
            self.drone.set_target_orientation(yaw, 0.0, -0.15)  # Slight forward pitch
            
        elif self.current_state == FlightState.ROTATE:
            # Use PID to execute a 180-degree rotation while maintaining position
            # Calculate target yaw angle for rotation (current + 180 degrees, normalized)
            current_yaw = orientation[2]
            target_yaw = current_yaw + np.pi  # 180-degree rotation
            
            # Normalize to -π to π range
            target_yaw = np.mod(target_yaw + np.pi, 2 * np.pi) - np.pi
            
            # Maintain position while rotating
            self.drone.set_target_position(position[0], position[1], position[2])
            self.drone.set_target_orientation(target_yaw, 0.0, 0.0)
            
        elif self.current_state == FlightState.DESCEND:
            # Use PID to control descent to 2m altitude
            target_altitude = 2.0
            
            # Maintain horizontal position, only change altitude
            self.drone.set_target_position(position[0], position[1], target_altitude)
            self.drone.set_target_orientation(orientation[2], 0.0, 0.0)
            
        elif self.current_state == FlightState.LAND:
            # Use PID for controlled landing
            # Target ground level (0m) with gentle descent
            target_altitude = 0.0
            
            # Maintain horizontal position, approach ground level
            self.drone.set_target_position(position[0], position[1], target_altitude)
            self.drone.set_target_orientation(orientation[2], 0.0, 0.0)
            
            # Override with direct thrust control for final landing phase
            # If we're close to the ground, use direct control for gentle touchdown
            if position[2] < 0.5:
                self.drone.set_thrust(max(0.0, 0.2 - position[2]/2))  # Reduce thrust near ground

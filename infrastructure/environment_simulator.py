"""
Environmental factors simulation for drone navigation
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class WindModel:
    """
    Models wind forces and turbulence in different weather conditions
    
    Attributes:
        base_wind_speed: Base wind speed in m/s
        wind_direction: Direction of the wind in radians
        turbulence_intensity: Intensity of turbulence (0.0 to 1.0)
        gust_probability: Probability of a wind gust in each update (0.0 to 1.0)
        gust_duration: Duration of gusts in simulation steps
        gust_max_strength: Maximum additional wind speed during a gust
    """
    
    def __init__(self, 
                 base_wind_speed: float = 0.0,
                 wind_direction: float = 0.0,
                 turbulence_intensity: float = 0.0,
                 gust_probability: float = 0.0,
                 gust_max_strength: float = 2.0,
                 gust_duration: int = 3):
        """
        Initialize the wind model with specified parameters
        
        Args:
            base_wind_speed: Base wind speed in m/s
            wind_direction: Direction of the wind in radians (0 = East, π/2 = North)
            turbulence_intensity: Intensity of turbulence (0.0 to 1.0)
            gust_probability: Probability of a wind gust in each update (0.0 to 1.0)
            gust_max_strength: Maximum additional wind speed during a gust
            gust_duration: Duration of gusts in simulation steps
        """
        self.base_wind_speed = base_wind_speed
        self.wind_direction = wind_direction
        self.turbulence_intensity = max(0.0, min(1.0, turbulence_intensity))
        self.gust_probability = max(0.0, min(1.0, gust_probability))
        self.gust_max_strength = gust_max_strength
        self.gust_duration = gust_duration
        
        # Internal state for gusts
        self._current_gust_strength = 0.0
        self._current_gust_timer = 0
        self._gust_active = False
        
        # Internal state for turbulence
        self._turbulence_state = np.zeros(3)
        
        logger.info(f"Wind model initialized: speed={base_wind_speed} m/s, " 
                   f"direction={np.degrees(wind_direction)}°, "
                   f"turbulence={turbulence_intensity}, "
                   f"gust_probability={gust_probability}")
    
    def update(self, timestep: int) -> np.ndarray:
        """
        Update the wind model and return the current wind vector
        
        Args:
            timestep: Current simulation timestep
            
        Returns:
            3D wind force vector (x, y, z) in m/s
        """
        # Base wind vector
        wind_vec = np.array([
            self.base_wind_speed * np.cos(self.wind_direction),
            self.base_wind_speed * np.sin(self.wind_direction),
            0.0  # Assuming horizontal wind only
        ])
        
        # Update gust state
        if self._gust_active:
            self._current_gust_timer -= 1
            if self._current_gust_timer <= 0:
                # Gust ended
                self._gust_active = False
                self._current_gust_strength = 0.0
                logger.debug("Wind gust ended")
        elif self.gust_probability > 0 and np.random.random() < self.gust_probability:
            # Start a new gust
            self._gust_active = True
            self._current_gust_strength = np.random.uniform(0.5, 1.0) * self.gust_max_strength
            self._current_gust_timer = self.gust_duration
            logger.debug(f"Wind gust started: strength={self._current_gust_strength} m/s, duration={self.gust_duration} steps")
        
        # Add gust effect to wind vector
        if self._gust_active:
            # Gust direction can vary slightly from base wind direction
            gust_direction = self.wind_direction + np.random.uniform(-0.2, 0.2)
            gust_vec = np.array([
                self._current_gust_strength * np.cos(gust_direction),
                self._current_gust_strength * np.sin(gust_direction),
                np.random.uniform(-0.5, 0.5) * self._current_gust_strength  # Small vertical component
            ])
            wind_vec += gust_vec
        
        # Update and add turbulence
        if self.turbulence_intensity > 0:
            # Simple turbulence model using filtered random noise
            # Update current turbulence state with new random input and filtering
            random_input = np.random.normal(0, 1, 3) * self.turbulence_intensity
            
            # Apply simple exponential filter for temporal correlation
            alpha = 0.7  # Filter coefficient (0 = no filter, 1 = no update)
            self._turbulence_state = alpha * self._turbulence_state + (1 - alpha) * random_input
            
            # Scale turbulence based on current wind speed and add to wind vector
            turbulence_scale = max(0.5, self.base_wind_speed)
            wind_vec += self._turbulence_state * turbulence_scale
        
        return wind_vec

class EnvironmentSimulator:
    """
    Simulates environmental factors affecting drone flight
    
    Includes wind, turbulence, and other atmospheric conditions.
    """
    
    def __init__(self, 
                 enable_wind: bool = True,
                 wind_scenario: str = 'calm'):
        """
        Initialize the environment simulator
        
        Args:
            enable_wind: Whether to enable wind simulation
            wind_scenario: Predefined wind scenario ('calm', 'light', 'moderate', 'strong', 'gusty')
        """
        self.enable_wind = enable_wind
        
        # Initialize wind model based on scenario
        if enable_wind:
            wind_params = self._get_wind_params(wind_scenario)
            self.wind_model = WindModel(**wind_params)
        else:
            self.wind_model = None
        
        logger.info(f"Environment simulator initialized with wind={'enabled' if enable_wind else 'disabled'}, "
                   f"scenario='{wind_scenario}'")
    
    def _get_wind_params(self, scenario: str) -> Dict[str, Any]:
        """
        Get wind parameters for a predefined scenario
        
        Args:
            scenario: Wind scenario name
            
        Returns:
            Dictionary of wind parameters
        """
        # Define scenarios with different wind conditions
        scenarios = {
            'calm': {
                'base_wind_speed': 0.5,
                'wind_direction': np.radians(45),
                'turbulence_intensity': 0.1,
                'gust_probability': 0.01,
                'gust_max_strength': 1.0,
                'gust_duration': 2
            },
            'light': {
                'base_wind_speed': 2.0,
                'wind_direction': np.radians(90),
                'turbulence_intensity': 0.2,
                'gust_probability': 0.05,
                'gust_max_strength': 3.0,
                'gust_duration': 3
            },
            'moderate': {
                'base_wind_speed': 4.0,
                'wind_direction': np.radians(0),
                'turbulence_intensity': 0.3,
                'gust_probability': 0.1,
                'gust_max_strength': 6.0,
                'gust_duration': 4
            },
            'strong': {
                'base_wind_speed': 8.0,
                'wind_direction': np.radians(270),
                'turbulence_intensity': 0.5,
                'gust_probability': 0.15,
                'gust_max_strength': 10.0,
                'gust_duration': 5
            },
            'gusty': {
                'base_wind_speed': 3.0,
                'wind_direction': np.radians(180),
                'turbulence_intensity': 0.4,
                'gust_probability': 0.3,
                'gust_max_strength': 8.0,
                'gust_duration': 3
            }
        }
        
        # Default to calm if scenario not found
        return scenarios.get(scenario.lower(), scenarios['calm'])
    
    def get_environmental_forces(self, timestep: int, drone_position: Tuple[float, float, float], 
                               drone_velocity: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
        """
        Get environmental forces acting on the drone at the current timestep
        
        Args:
            timestep: Current simulation timestep
            drone_position: Current drone position (x, y, z) in meters
            drone_velocity: Current drone velocity (vx, vy, vz) in m/s
            
        Returns:
            Dictionary of force vectors (wind, etc.)
        """
        forces = {}
        
        # Calculate wind force if enabled
        if self.enable_wind and self.wind_model:
            wind_vector = self.wind_model.update(timestep)
            
            # Adjust wind based on altitude (simplified model)
            # Wind typically increases with altitude
            altitude = drone_position[2]
            if altitude > 0:
                # Simple logarithmic wind profile
                altitude_factor = max(1.0, 1.0 + 0.2 * np.log10(1 + altitude))
                wind_vector = wind_vector * altitude_factor
            
            # Calculate relative wind (wind - drone_velocity)
            # This affects the aerodynamic forces on the drone
            relative_wind = wind_vector - np.array(drone_velocity)
            
            forces['wind'] = wind_vector
            forces['relative_wind'] = relative_wind
            
            # Simple drag force model based on relative wind
            drag_coefficient = 0.4  # Simplified drag coefficient
            drone_cross_section = 0.1  # Simplified cross-sectional area in m²
            air_density = 1.225  # kg/m³ at sea level
            
            # Drag force = 0.5 * rho * v² * Cd * A
            wind_speed_squared = np.sum(relative_wind**2)
            drag_magnitude = 0.5 * air_density * wind_speed_squared * drag_coefficient * drone_cross_section
            
            # Drag direction is opposite to relative wind
            if np.linalg.norm(relative_wind) > 0:
                drag_direction = -relative_wind / np.linalg.norm(relative_wind)
                drag_force = drag_direction * drag_magnitude
            else:
                drag_force = np.zeros(3)
            
            forces['drag'] = drag_force
        
        return forces
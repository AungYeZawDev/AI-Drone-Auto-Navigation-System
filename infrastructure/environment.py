"""
Environmental factors for drone simulation
"""
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WindModel:
    """
    Simulates wind and turbulence in the environment
    
    Generates realistic wind vectors with direction, intensity, and gustiness
    characteristics that change over time.
    """
    
    def __init__(self, 
                 base_velocity: float = 2.0,
                 direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 turbulence_intensity: float = 0.5,
                 gust_probability: float = 0.1,
                 max_gust_strength: float = 5.0,
                 wind_scenario: str = "moderate"):
        """
        Initialize the wind model
        
        Args:
            base_velocity: Base wind speed in m/s
            direction: Base wind direction vector (will be normalized)
            turbulence_intensity: How much the wind varies (0.0 to 1.0)
            gust_probability: Probability of a gust occurring at each step
            max_gust_strength: Maximum strength of gusts
            wind_scenario: Predefined weather scenario (overrides individual params)
        """
        # Check if using a predefined scenario
        if wind_scenario and wind_scenario != "custom":
            scenario_params = self._get_scenario_parameters(wind_scenario)
            base_velocity = scenario_params.get('base_velocity', base_velocity)
            direction = scenario_params.get('direction', direction)
            turbulence_intensity = scenario_params.get('turbulence_intensity', turbulence_intensity)
            gust_probability = scenario_params.get('gust_probability', gust_probability)
            max_gust_strength = scenario_params.get('max_gust_strength', max_gust_strength)
        
        self.base_velocity = base_velocity
        
        # Normalize direction vector
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            self.direction = np.array(direction) / direction_norm
        else:
            self.direction = np.array([1.0, 0.0, 0.0])  # Default to east wind
            
        self.turbulence_intensity = max(0.0, min(1.0, turbulence_intensity))
        self.gust_probability = max(0.0, min(1.0, gust_probability))
        self.max_gust_strength = max_gust_strength
        
        # Current wind state
        self.current_velocity = base_velocity
        self.current_direction = self.direction.copy()
        self.current_gust_remaining = 0  # Timesteps remaining for current gust
        
        # Advanced turbulence modeling
        self.turbulence_filter = 0.7  # Controls how rapidly turbulence changes (0-1)
        self.turbulence_state = np.zeros(3)  # Current turbulence state
        
        # Altitude-based effects
        self.altitude_effect_factor = 0.1  # How much altitude affects wind
        
        # Wind history for visualization
        self.wind_history: List[Dict[str, Any]] = []
        
        # Random seed for reproducibility in testing
        self.seed = np.random.randint(0, 10000)
        
        logger.info(f"Wind model initialized with scenario '{wind_scenario}', base velocity {base_velocity}m/s")
        
    def _get_scenario_parameters(self, scenario: str) -> Dict[str, Any]:
        """
        Get predefined parameters for different weather scenarios
        
        Args:
            scenario: Weather scenario name
            
        Returns:
            Dictionary of wind parameters
        """
        # Define different weather scenarios
        scenarios = {
            "calm": {
                "base_velocity": 0.5,
                "direction": (1.0, 0.0, 0.0),  # East
                "turbulence_intensity": 0.1,
                "gust_probability": 0.01,
                "max_gust_strength": 1.0
            },
            "light": {
                "base_velocity": 2.0,
                "direction": (0.0, 1.0, 0.0),  # North
                "turbulence_intensity": 0.2,
                "gust_probability": 0.05,
                "max_gust_strength": 3.0
            },
            "moderate": {
                "base_velocity": 4.0,
                "direction": (1.0, 1.0, 0.0),  # Northeast
                "turbulence_intensity": 0.4,
                "gust_probability": 0.1,
                "max_gust_strength": 6.0
            },
            "strong": {
                "base_velocity": 8.0,
                "direction": (0.0, -1.0, 0.0),  # South
                "turbulence_intensity": 0.6,
                "gust_probability": 0.15,
                "max_gust_strength": 12.0
            },
            "stormy": {
                "base_velocity": 12.0,
                "direction": (-1.0, -1.0, 0.0),  # Southwest
                "turbulence_intensity": 0.8,
                "gust_probability": 0.25,
                "max_gust_strength": 20.0
            },
            "gusty": {
                "base_velocity": 3.0,
                "direction": (-1.0, 0.0, 0.0),  # West
                "turbulence_intensity": 0.5,
                "gust_probability": 0.3,
                "max_gust_strength": 10.0
            }
        }
        
        # Return default if scenario not found
        return scenarios.get(scenario.lower(), scenarios["moderate"])
        
    def update(self, timestep: int, altitude: float = 0.0) -> np.ndarray:
        """
        Update wind conditions for the current timestep
        
        Args:
            timestep: Current simulation timestep
            altitude: Current altitude in meters (for altitude-dependent effects)
            
        Returns:
            Wind force vector [Fx, Fy, Fz] in Newtons
        """
        # Set random seed for reproducibility if testing
        # np.random.seed(self.seed + timestep)
        
        # ==== Advanced Turbulence Modeling ====
        # Generate new random turbulence input
        turbulence_input = np.random.normal(0, self.turbulence_intensity, 3)
        
        # Apply exponential filtering for temporal correlation (smoother changes)
        self.turbulence_state = self.turbulence_filter * self.turbulence_state + \
                              (1 - self.turbulence_filter) * turbulence_input
        
        # Apply turbulence to velocity and direction
        velocity_turbulence = self.turbulence_state[0] * self.base_velocity * 0.3
        direction_turbulence = self.turbulence_state[1:3] * 0.15
        
        # ==== Gust Modeling ====
        # Check if current gust has ended
        if self.current_gust_remaining > 0:
            self.current_gust_remaining -= 1
            # Gradually reduce gust strength toward the end for smoother transition
            if self.current_gust_remaining == 1:
                # Taper off the gust
                self.current_velocity = self.base_velocity + velocity_turbulence
        else:
            self.current_velocity = self.base_velocity + velocity_turbulence
        
        # Check for new gust
        if self.current_gust_remaining == 0 and np.random.random() < self.gust_probability:
            # Generate more realistic gust profile
            gust_strength = np.random.uniform(0.5, self.max_gust_strength)
            gust_duration = np.random.randint(2, 8)  # 2-7 timesteps
            
            # Randomize gust direction slightly
            gust_dir_change = np.random.normal(0, 0.2, 3)
            
            self.current_velocity += gust_strength
            self.current_gust_remaining = gust_duration
            
            # Apply small direction change during gust
            gust_direction = self.direction + gust_dir_change
            gust_direction_norm = np.linalg.norm(gust_direction)
            if gust_direction_norm > 0:
                self.current_direction = gust_direction / gust_direction_norm
            
            logger.info(f"Wind gust at t={timestep}: {gust_strength:.2f}m/s for {gust_duration} steps")
        
        # ==== Direction Updates ====
        # Create direction variation with memory (temporal correlation)
        direction_variation = np.array([
            direction_turbulence[0], 
            direction_turbulence[1],
            0.1 * self.turbulence_state[2]  # Small vertical component
        ])
        
        # Update direction with turbulence and normalize
        self.current_direction = self.direction + direction_variation
        direction_norm = np.linalg.norm(self.current_direction)
        if direction_norm > 0:
            self.current_direction = self.current_direction / direction_norm
        
        # ==== Altitude Effects ====
        # Apply altitude-dependent scaling (wind typically increases with altitude)
        if altitude > 0:
            # Logarithmic wind profile (common in atmospheric boundary layer)
            # v(z) = v_ref * ln(z/z0) / ln(z_ref/z0)
            # Simplified version:
            z0 = 0.1  # roughness length (m)
            z_ref = 10.0  # reference height (m)
            if altitude < z0:
                altitude_factor = 0.5  # Minimum factor near ground
            else:
                altitude_factor = 1.0 + self.altitude_effect_factor * np.log(1 + altitude / z0)
                
            # Cap the maximum factor
            altitude_factor = min(altitude_factor, 3.0)
            
            # Apply to velocity
            velocity_with_altitude = self.current_velocity * altitude_factor
        else:
            # Below ground level (shouldn't happen in normal simulation)
            velocity_with_altitude = self.current_velocity * 0.5
        
        # ==== Force Calculation ====
        # Calculate wind force vector using aerodynamic principles
        # Force = 0.5 * rho * v^2 * Cd * A
        air_density = 1.225  # kg/m³ at sea level
        
        # Altitude affects air density (simplified model)
        if altitude > 0:
            # Standard atmosphere model (simplified)
            air_density *= np.exp(-altitude / 8000)  # Scale height ~8000m
        
        drag_coefficient = 1.2  # More realistic drag coefficient for drone
        effective_area = 0.15  # m² (typical drone cross-section)
        
        # Calculate force magnitude
        force_magnitude = 0.5 * air_density * velocity_with_altitude**2 * drag_coefficient * effective_area
        
        # Create final wind force vector
        wind_force = force_magnitude * self.current_direction
        
        # Record history for visualization
        self.wind_history.append({
            'timestep': timestep,
            'velocity': velocity_with_altitude,
            'direction': self.current_direction.copy(),
            'force': wind_force.copy(),
            'altitude': altitude,
            'turbulence': self.turbulence_state.copy(),
            'gust_remaining': self.current_gust_remaining
        })
        
        return wind_force
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get wind history for visualization
        
        Returns:
            List of wind state dictionaries
        """
        return self.wind_history

class Environment:
    """
    Models the physical environment for the drone simulation
    
    Includes wind, temperature, pressure variations, and other
    environmental factors that affect drone flight.
    """
    
    def __init__(self, wind_scenario: str = "moderate", enable_wind: bool = True):
        """
        Initialize the environment model
        
        Args:
            wind_scenario: Weather scenario to use ("calm", "light", "moderate", "strong", "stormy", "gusty")
            enable_wind: Whether to enable wind simulation
        """
        self.enable_wind = enable_wind
        self.wind_scenario = wind_scenario
        
        # Initialize wind model with chosen scenario
        if enable_wind:
            self.wind_model = WindModel(wind_scenario=wind_scenario)
        else:
            # Create a minimal wind model with negligible effects
            self.wind_model = WindModel(wind_scenario="calm")
            self.wind_model.base_velocity = 0.01
            self.wind_model.turbulence_intensity = 0.01
            self.wind_model.gust_probability = 0.0
        
        # Default environment parameters
        self.temperature = 20.0  # °C
        self.pressure = 101325  # Pa (standard atmosphere)
        self.gravity = 9.81  # m/s²
        
        # Weather effects on temperature and pressure
        if wind_scenario == "stormy":
            self.temperature -= 5.0  # Colder during storms
            self.pressure -= 2000  # Lower pressure in storms
        elif wind_scenario == "strong":
            self.temperature -= 2.0
            self.pressure -= 1000
        
        # List to store environmental data history
        self.environment_history: List[Dict[str, Any]] = []
        
        logger.info(f"Environment model initialized with wind scenario '{wind_scenario}', " +
                   f"wind {'enabled' if enable_wind else 'disabled'}")
    
    def update(self, timestep: int, drone_position: Tuple[float, float, float] = (0, 0, 0),
              drone_velocity: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """
        Update environmental conditions for the current timestep
        
        Args:
            timestep: Current simulation timestep
            drone_position: Current drone position (x, y, z)
            drone_velocity: Current drone velocity (vx, vy, vz)
            
        Returns:
            Dictionary of environmental forces and conditions
        """
        # Get altitude from drone position
        altitude = drone_position[2]
        
        # Get wind forces, adjusted for altitude
        wind_force = self.wind_model.update(timestep, altitude=altitude)
        
        # Calculate relative wind (wind - drone_velocity)
        # This is important for realistic aerodynamic forces
        if self.enable_wind:
            relative_velocity = np.array(wind_force) - np.array(drone_velocity)
            
            # Calculate drag due to relative wind
            # Drag = 0.5 * rho * v^2 * Cd * A
            air_density = 1.225  # kg/m³ at sea level
            
            # Adjust density with altitude
            if altitude > 0:
                air_density *= np.exp(-altitude / 8000)  # Scale height ~8000m
                
            drag_coefficient = 0.5  # Simplified drag coefficient
            effective_area = 0.1  # m² (approximate cross-section)
            
            # Drag force magnitude
            rel_velocity_magnitude = np.linalg.norm(relative_velocity)
            if rel_velocity_magnitude > 0:
                drag_magnitude = 0.5 * air_density * rel_velocity_magnitude**2 * drag_coefficient * effective_area
                drag_direction = -relative_velocity / rel_velocity_magnitude
                drag_force = drag_direction * drag_magnitude
            else:
                drag_force = np.zeros(3)
        else:
            # No wind, no drag
            relative_velocity = np.zeros(3)
            drag_force = np.zeros(3)
        
        # Calculate air pressure based on altitude (barometric formula)
        # P = P₀ * exp(-g * M * h / (R * T))
        if altitude > 0:
            # Constants
            g = self.gravity  # m/s²
            M = 0.02896  # kg/mol (molar mass of air)
            R = 8.314  # J/(mol·K) (universal gas constant)
            T = self.temperature + 273.15  # K (convert from °C)
            
            # Calculate pressure at altitude
            altitude_pressure = self.pressure * np.exp(-g * M * altitude / (R * T))
        else:
            altitude_pressure = self.pressure
        
        # Temperature decreases with altitude (lapse rate)
        lapse_rate = 0.0065  # K/m (standard atmospheric lapse rate)
        altitude_temperature = max(-40, self.temperature - lapse_rate * max(0, altitude))
        
        # Collect all environmental data
        env_data = {
            'wind_force': wind_force,
            'relative_wind': relative_velocity,
            'drag_force': drag_force,
            'temperature': altitude_temperature,
            'pressure': altitude_pressure,
            'gravity': self.gravity,
            'altitude': altitude
        }
        
        # Record history
        self.environment_history.append({
            'timestep': timestep,
            **env_data
        })
        
        return env_data
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get environment history for visualization
        
        Returns:
            List of environment state dictionaries
        """
        return self.environment_history
import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")

# Default configuration values
default_config = {
    "environment_config": {
        "wind_scenario": "moderate",
        "enable_wind": True
    },
    "drone_config": {
        "pid_gains": {
            "altitude": {"kp": 1.0, "ki": 0.1, "kd": 0.4},
            "position_x": {"kp": 0.25, "ki": 0.015, "kd": 0.07},
            "position_y": {"kp": 0.25, "ki": 0.015, "kd": 0.07},
            "yaw": {"kp": 0.7, "ki": 0.01, "kd": 0.1},
            "roll": {"kp": 0.6, "ki": 0.0, "kd": 0.1},
            "pitch": {"kp": 0.6, "ki": 0.0, "kd": 0.1}
        }
    }
}

# Serve the index page
@app.route('/')
def index():
    """Serve the main page of the drone simulation interface"""
    return render_template('index.html')

# API endpoints for simulation control
@app.route('/api/simulation/start', methods=['POST'])
def start_simulation_api():
    """Start a new drone simulation with the specified parameters"""
    try:
        # Import the simulation module here to avoid circular imports
        import main as sim
        
        data = request.json or {}
        
        # Prepare configuration
        config = {
            'duration': 60,  # seconds
            'update_rate': 1,  # Hz
            'environment': {},
            'drone': {'pid_gains': {}}
        }
        
        # Update environment config if provided
        if 'environment' in data:
            config['environment'].update(data['environment'])
        else:
            config['environment'].update(default_config['environment_config'])
        
        # Update drone config if provided
        if 'drone' in data and 'pid_gains' in data['drone']:
            config['drone']['pid_gains'].update(data['drone']['pid_gains'])
        else:
            config['drone']['pid_gains'].update(default_config['drone_config']['pid_gains'])
        
        # Start simulation in a separate thread
        sim.start_simulation_thread(config)
        
        # Return initial simulation state
        return jsonify({
            'status': 'started',
            'config': config
        })
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation_api():
    """Stop the currently running simulation"""
    try:
        # Import the simulation module here to avoid circular imports
        import main as sim
        
        # Stop the simulation
        sim.stop_simulation()
        
        return jsonify({'status': 'stopped'})
    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get the current status of the simulation"""
    try:
        # Import the simulation module here to avoid circular imports
        import main as sim
        
        # Get current simulation status
        status = sim.get_simulation_status()
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting simulation status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/drone/command', methods=['POST'])
def send_drone_command():
    """Send a command to the drone"""
    try:
        command = request.json
        
        # Handle different command types
        if command.get('type') == 'update_pid':
            logger.info(f"Received PID update: {command.get('settings')}")
            # In a full implementation, this would update the PID controllers in the simulation
            
        return jsonify({'status': 'command_sent', 'command': command})
    except Exception as e:
        logger.error(f"Error sending drone command: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plots', methods=['GET'])
def get_available_plots():
    """Get a list of available plot files"""
    try:
        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        plot_files = [f for f in os.listdir('plots') if f.endswith('.png')]
        return jsonify({'plots': plot_files})
    except Exception as e:
        logger.error(f"Error getting available plots: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve a plot file"""
    return send_from_directory('plots', filename)

@app.route('/api/environment/wind_scenarios', methods=['GET'])
def get_wind_scenarios():
    """Get available wind scenarios"""
    scenarios = [
        {"id": "calm", "name": "Calm", "description": "Almost no wind (0-1 m/s)"},
        {"id": "light", "name": "Light", "description": "Light breeze (1-3 m/s)"},
        {"id": "moderate", "name": "Moderate", "description": "Moderate wind (3-5 m/s)"},
        {"id": "strong", "name": "Strong", "description": "Strong wind (5-8 m/s)"},
        {"id": "stormy", "name": "Stormy", "description": "Storm conditions (8-12 m/s)"},
        {"id": "gusty", "name": "Gusty", "description": "Moderate wind with frequent gusts"}
    ]
    return jsonify({"scenarios": scenarios})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
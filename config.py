"""
Configuration module for Hydroponic ML Project
Manages all hyperparameters and settings
"""

import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    """Configuration class for the project"""
    
    # Dataset Configuration
    DATASET_PATH: str = 'data/IoTData-Raw.csv'
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    # Data Preprocessing
    SCALER_TYPE: str = 'standard'  # 'standard' or 'minmax'
    SENSOR_FEATURES: list = None
    CONTROL_FEATURES: list = None
    TARGET_VARIABLE: str = 'system_health'  # Will be engineered from sensor data
    
    # CNN Architecture
    CNN_INPUT_SHAPE: tuple = (80,)  # For 1D sensor data
    CNN_FILTERS: list = None
    CNN_KERNELS: list = None
    CNN_POOL_SIZE: int = 2
    CNN_DENSE_UNITS: list = None
    CNN_DROPOUT_RATE: float = 0.3
    CNN_ACTIVATION: str = 'relu'
    CNN_OUTPUT_ACTIVATION: str = 'sigmoid'
    
    # Training Parameters
    BASELINE_EPOCHS: int = 50
    OPTIMIZED_EPOCHS: int = 50
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    OPTIMIZER: str = 'adam'
    LOSS_FUNCTION: str = 'binary_crossentropy'
    METRICS: list = None
    EARLY_STOPPING_PATIENCE: int = 10
    VALIDATION_SPLIT: float = 0.2
    
    # PSO Configuration
    PSO_PARTICLES: int = 5
    PSO_ITERATIONS: int = 10
    PSO_INERTIA_MIN: float = 0.4
    PSO_INERTIA_MAX: float = 0.9
    PSO_COGNITIVE: float = 2.0
    PSO_SOCIAL: float = 2.0
    PSO_TOPOLOGY: str = 'global'  # 'global' or 'local'
    
    # Hyperparameter Search Space (for PSO)
    HYPERPARAM_SPACE: Dict[str, tuple] = None
    
    # Output Configuration
    OUTPUT_DIR: str = 'results'
    MODELS_DIR: str = 'models'
    LOGS_DIR: str = 'logs'
    SAVE_PLOTS: bool = True
    PLOT_FORMAT: str = 'png'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    VERBOSE: int = 1
    
    def __init__(self, config_file=None):
        """Initialize configuration with defaults and optional file"""
        self._set_defaults()
        
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Create output directories
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
    
    def _set_defaults(self):
        """Set default values for all parameters"""
        # Default CNN architecture
        self.CNN_FILTERS = [32, 64, 128]
        self.CNN_KERNELS = [3, 3, 3]
        self.CNN_DENSE_UNITS = [256, 128, 64]
        
        # Default sensor features from IoT data
        self.SENSOR_FEATURES = [
            'pH', 'TDS', 'water_level', 'DHT_temp', 
            'DHT_humidity', 'water_temp'
        ]
        
        # Default control features
        self.CONTROL_FEATURES = [
            'pH_reducer', 'add_water', 'nutrients_adder',
            'humidifier', 'ex_fan'
        ]
        
        # Default metrics
        self.METRICS = ['accuracy', 'precision', 'recall', 'auc']
        
        # Default hyperparameter search space for PSO
        self.HYPERPARAM_SPACE = {
            'learning_rate': (0.0001, 0.01),
            'batch_size': (16, 64),
            'dropout_rate': (0.1, 0.5),
            'dense_units_1': (64, 256),
            'dense_units_2': (32, 128),
            'dense_units_3': (16, 64),
        }
    
    def _load_from_file(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
                
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        for key, value in self.to_dict().items():
            if not isinstance(value, (list, dict)):
                print(f"{key:30s}: {value}")
        print("="*80 + "\n")


# Default configuration instance
default_config = Config()

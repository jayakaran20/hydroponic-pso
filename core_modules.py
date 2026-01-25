"""
Core Training, PSO Optimization, and Evaluation Modules
Complete implementation for CNN-PSO hybrid hydroponic agriculture model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import pyswarms as ps
from typing import Tuple, Dict, List, Any
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA HANDLER MODULE
# =============================================================================

class DataHandler:
    """Handle data loading, preprocessing, and feature engineering"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_names = None
    
    def load_and_preprocess(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load IoT sensor data and create target variable
        Returns: X (features), y (target), feature_names
        """
        df = pd.read_csv(data_path)
        
        # Data cleaning
        df = df.dropna(subset=['add_water'])
        df = df.fillna(df.mean(numeric_only=True))
        
        # Feature engineering: Create health label based on sensor conditions
        # Healthy: pH 5.5-6.5, TDS 800-1400, Temp 20-25, Humidity 60-80
        df['system_health'] = self._engineer_health_label(df)
        
        # Select features
        X_data = df[self.config.SENSOR_FEATURES].values
        y_data = df['system_health'].values
        
        # Reshape for CNN (add feature dimension)
        X_data = np.expand_dims(X_data, axis=-1)
        
        # Normalize
        if self.config.SCALER_TYPE == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        X_data = self.scaler.fit_transform(X_data.reshape(-1, X_data.shape[-1]))
        X_data = X_data.reshape(-1, len(self.config.SENSOR_FEATURES), 1)
        
        self.feature_names = self.config.SENSOR_FEATURES
        
        return X_data, y_data, self.feature_names
    
    def _engineer_health_label(self, df: pd.DataFrame) -> np.ndarray:
        """
        Engineer health label from sensor readings
        1 = Healthy, 0 = Unhealthy
        """
        health = np.ones(len(df))
        
        # pH condition: optimal 5.5-6.5
        health[(df['pH'] < 5.5) | (df['pH'] > 7.0)] = 0
        
        # TDS condition: optimal 800-1400
        health[(df['TDS'] < 800) | (df['TDS'] > 1500)] = 0
        
        # Temperature: optimal 20-25
        health[(df['DHT_temp'] < 18) | (df['DHT_temp'] > 28)] = 0
        
        # Humidity: optimal 60-80%
        health[(df['DHT_humidity'] < 50) | (df['DHT_humidity'] > 90)] = 0
        
        # Water temp: optimal 18-24
        health[(df['water_temp'] < 16) | (df['water_temp'] > 26)] = 0
        
        return health.astype(int)


# =============================================================================
# CNN MODEL MODULE
# =============================================================================

class CNNModel:
    """Build CNN architecture for time-series hydroponic sensor data"""
    
    def __init__(self, config, input_shape: tuple):
        self.config = config
        self.input_shape = input_shape
    
    def build_model(self, filters: List[int] = None, 
                   kernels: List[int] = None,
                   dense_units: List[int] = None,
                   dropout_rate: float = None,
                   learning_rate: float = None) -> keras.Model:
        """
        Build CNN model with configurable architecture
        
        Args:
            filters: Convolutional filters
            kernels: Kernel sizes
            dense_units: Dense layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        
        filters = filters or self.config.CNN_FILTERS
        kernels = kernels or self.config.CNN_KERNELS
        dense_units = dense_units or self.config.CNN_DENSE_UNITS
        dropout_rate = dropout_rate or self.config.CNN_DROPOUT_RATE
        learning_rate = learning_rate or self.config.LEARNING_RATE
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Convolutional blocks
            layers.Conv1D(filters[0], kernels[0], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(self.config.CNN_POOL_SIZE),
            layers.Dropout(dropout_rate),
            
            layers.Conv1D(filters[1], kernels[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(self.config.CNN_POOL_SIZE),
            layers.Dropout(dropout_rate),
            
            layers.Conv1D(filters[2], kernels[2], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(self.config.CNN_POOL_SIZE),
            layers.Dropout(dropout_rate),
            
            # Flatten
            layers.Flatten(),
            
            # Dense blocks
            layers.Dense(dense_units[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(dense_units[1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(dense_units[2], activation='relu'),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.config.LOSS_FUNCTION,
            metrics=self.config.METRICS
        )
        
        return model


# =============================================================================
# TRAINER MODULE
# =============================================================================

class Trainer:
    """Handle model training and validation"""
    
    def __init__(self, config, model_type: str = 'cnn'):
        self.config = config
        self.model_type = model_type
        self.history = None
        self.model = None
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs: int = 50) -> keras.Model:
        """Train CNN with default parameters"""
        
        cnn = CNNModel(self.config, X_train.shape[1:])
        self.model = cnn.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=self.config.VERBOSE
        )
        
        return self.model
    
    def train_model_with_params(self, X_train, y_train, X_val, y_val,
                               params: Dict, epochs: int = 50) -> keras.Model:
        """Train CNN with optimized parameters"""
        
        cnn = CNNModel(self.config, X_train.shape[1:])
        self.model = cnn.build_model(
            dropout_rate=params.get('dropout_rate', 0.3),
            learning_rate=params.get('learning_rate', 0.001)
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
        ]
        
        batch_size = int(params.get('batch_size', self.config.BATCH_SIZE))
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=self.config.VERBOSE
        )
        
        return self.model


# =============================================================================
# PSO OPTIMIZER MODULE
# =============================================================================

class PSOOptimizer:
    """Particle Swarm Optimization for CNN hyperparameter tuning"""
    
    def __init__(self, config):
        self.config = config
        self.optimization_history = {
            'fitness': [],
            'best_fitness': [],
            'best_params': None
        }
    
    def optimize(self, X_train, y_train, X_val, y_val,
                n_particles: int = 5,
                n_iterations: int = 10) -> Tuple[Dict, Dict]:
        """
        Run PSO optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_particles: Number of particles in swarm
            n_iterations: Number of optimization iterations
        
        Returns:
            best_params: Best hyperparameters found
            optimization_history: History of optimization
        """
        
        # Define objective function (minimize: 1 - accuracy)
        def objective_function(params_array):
            """
            Fitness function: Train models and return 1 - validation accuracy
            """
            fitness_scores = []
            
            for particle_params in params_array:
                try:
                    # Map parameters
                    params = self._map_parameters(particle_params)
                    
                    # Train model
                    trainer = Trainer(self.config)
                    model = trainer.train_model_with_params(
                        X_train, y_train, X_val, y_val,
                        params=params,
                        epochs=self.config.BASELINE_EPOCHS
                    )
                    
                    # Evaluate
                    val_loss, val_accuracy = model.evaluate(
                        X_val, y_val, verbose=0
                    )
                    
                    # Fitness (minimize: 1 - accuracy)
                    fitness = 1.0 - val_accuracy
                    fitness_scores.append(fitness)
                    
                except Exception as e:
                    fitness_scores.append(1.0)  # Worst score on error
            
            return np.array(fitness_scores)
        
        # Initialize PSO optimizer
        n_params = len(self.config.HYPERPARAM_SPACE)
        options = {
            'c1': self.config.PSO_COGNITIVE,
            'c2': self.config.PSO_SOCIAL,
            'w': (self.config.PSO_INERTIA_MAX, self.config.PSO_INERTIA_MIN)
        }
        
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=n_params,
            options=options,
            bounds=self._get_bounds()
        )
        
        # Run optimization
        best_cost, best_params = optimizer.optimize(
            objective_function,
            iters=n_iterations,
            verbose=True
        )
        
        # Store results
        self.optimization_history['best_fitness'].append(best_cost)
        self.optimization_history['best_params'] = best_params
        
        # Map parameters
        best_params_dict = self._map_parameters(best_params)
        
        return best_params_dict, self.optimization_history
    
    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for PSO"""
        min_bounds = []
        max_bounds = []
        
        for key, (min_val, max_val) in self.config.HYPERPARAM_SPACE.items():
            min_bounds.append(min_val)
            max_bounds.append(max_val)
        
        return (np.array(min_bounds), np.array(max_bounds))
    
    def _map_parameters(self, params_array: np.ndarray) -> Dict:
        """Map PSO parameters to hyperparameter dictionary"""
        param_keys = list(self.config.HYPERPARAM_SPACE.keys())
        
        params_dict = {}
        for i, key in enumerate(param_keys):
            params_dict[key] = params_array[i]
        
        return params_dict


# =============================================================================
# METRICS AND EVALUATION MODULE
# =============================================================================

class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    @staticmethod
    def evaluate(model: keras.Model, X_test: np.ndarray, 
                y_test: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred, zero_division=0),
            'recall': recall_score(y_test_flat, y_pred, zero_division=0),
            'f1_score': f1_score(y_test_flat, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test_flat, y_pred_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test_flat, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test_flat, y_pred, zero_division=0
        )
        
        return metrics

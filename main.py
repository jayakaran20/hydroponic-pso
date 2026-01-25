"""
Main entry point for Hydroponic Agriculture ML Project
Hybrid Metaheuristic Optimization of Deep Learning Models
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data_handler import DataHandler
from models.cnn_model import CNNModel
from models.vgg_model import VGGModel
from optimization.pso_optimizer import PSOOptimizer
from training.trainer import Trainer
from evaluation.metrics import MetricsCalculator
from evaluation.visualization import Visualizer
from evaluation.comparison import ComparisonAnalysis
from utils.logger import Logger

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class HydroponicMLPipeline:
    """
    Complete pipeline for hybrid metaheuristic optimization of deep learning
    models for hydroponic agriculture using CNN-PSO hybrid approach
    """
    
    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.logger = Logger('HydroponicMLPipeline')
        self.results = {}
        
    def run_full_pipeline(self, data_path, mode='pso'):
        """
        Execute complete pipeline:
        1. Data loading and preprocessing
        2. Baseline CNN training
        3. PSO optimization (optional)
        4. Model evaluation and comparison
        5. Visualization and reporting
        """
        self.logger.info("="*80)
        self.logger.info("HYBRID METAHEURISTIC OPTIMIZATION FOR HYDROPONIC AGRICULTURE")
        self.logger.info("="*80)
        
        # Step 1: Load and preprocess data
        self.logger.info("\n[STEP 1] Loading and preprocessing data...")
        data_handler = DataHandler(self.config)
        X, y, feature_names = data_handler.load_and_preprocess(data_path)
        self.logger.info(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Step 2: Train baseline CNN
        self.logger.info("\n[STEP 2] Training baseline CNN...")
        baseline_trainer = Trainer(self.config, model_type='cnn')
        baseline_model = baseline_trainer.train_model(
            X_train, y_train, X_val, y_val, 
            epochs=self.config.BASELINE_EPOCHS
        )
        self.results['baseline_model'] = baseline_model
        
        # Evaluate baseline
        baseline_metrics = MetricsCalculator.evaluate(
            baseline_model, X_test, y_test
        )
        self.logger.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
        self.results['baseline_metrics'] = baseline_metrics
        
        # Step 3: PSO Optimization (if requested)
        if mode.lower() == 'pso':
            self.logger.info("\n[STEP 3] Running PSO optimization...")
            pso_optimizer = PSOOptimizer(self.config)
            best_params, optimization_history = pso_optimizer.optimize(
                X_train, y_train, X_val, y_val,
                n_particles=self.config.PSO_PARTICLES,
                n_iterations=self.config.PSO_ITERATIONS
            )
            self.logger.info(f"Best PSO Parameters: {best_params}")
            self.results['best_params'] = best_params
            self.results['optimization_history'] = optimization_history
            
            # Train optimized model with best parameters
            self.logger.info("\n[STEP 4] Training optimized CNN...")
            optimized_trainer = Trainer(self.config, model_type='cnn')
            optimized_model = optimized_trainer.train_model_with_params(
                X_train, y_train, X_val, y_val,
                params=best_params,
                epochs=self.config.OPTIMIZED_EPOCHS
            )
            self.results['optimized_model'] = optimized_model
            
            # Evaluate optimized model
            optimized_metrics = MetricsCalculator.evaluate(
                optimized_model, X_test, y_test
            )
            self.logger.info(f"Optimized Accuracy: {optimized_metrics['accuracy']:.4f}")
            self.results['optimized_metrics'] = optimized_metrics
        
        # Step 5: Comprehensive evaluation and visualization
        self.logger.info("\n[STEP 5] Generating comprehensive evaluation reports...")
        visualizer = Visualizer(self.config)
        
        # Create comparison analysis
        comparison = ComparisonAnalysis(self.config)
        if mode.lower() == 'pso':
            comparison_results = comparison.compare_models(
                baseline_metrics, optimized_metrics,
                optimization_history, best_params
            )
        else:
            comparison_results = {'baseline': baseline_metrics}
        
        self.results['comparison'] = comparison_results
        
        # Generate visualizations
        visualizer.plot_training_history(self.results)
        visualizer.plot_metrics_comparison(comparison_results, mode)
        if mode.lower() == 'pso':
            visualizer.plot_pso_convergence(optimization_history)
        visualizer.plot_confusion_matrices(
            baseline_model, X_test, y_test,
            optimized_model if mode.lower() == 'pso' else None
        )
        
        # Step 6: Save results
        self.logger.info("\n[STEP 6] Saving results and models...")
        self.save_results(baseline_model, 
                         optimized_model if mode.lower() == 'pso' else None)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        self.logger.info("="*80)
        
        return self.results
    
    def save_results(self, baseline_model, optimized_model=None):
        """Save models, metrics, and results to disk"""
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Save models
        baseline_model.save('models/baseline_cnn.h5')
        if optimized_model:
            optimized_model.save('models/pso_optimized_cnn.h5')
        
        # Save metrics
        with open('results/metrics_comparison.json', 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        
        self.logger.info("Models and results saved successfully")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Hybrid Metaheuristic Optimization for Hydroponic Agriculture'
    )
    parser.add_argument('--data', type=str, default='data/IoTData-Raw.csv',
                       help='Path to IoT sensor data')
    parser.add_argument('--mode', type=str, default='pso', 
                       choices=['baseline', 'pso'],
                       help='Execution mode: baseline CNN or PSO optimization')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # GPU setup
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Run pipeline
    pipeline = HydroponicMLPipeline(args.config)
    results = pipeline.run_full_pipeline(args.data, mode=args.mode)
    
    return results


if __name__ == '__main__':
    main()

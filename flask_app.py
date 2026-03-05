"""
Flask Web Application and REST API
Frontend interface for Hydroponic ML Project
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import os
import importlib.util
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

if importlib.util.find_spec('tensorflow'):
    import tensorflow as tf
else:
    tf = None

# Initialize Flask app
# NOTE: this project keeps `index.html` at repo root (not in `templates/`).
# Point Flask templates there so `/` works consistently in local runs and containers.
app = Flask(__name__, template_folder='.')
CORS(app)

# Global variables for models
baseline_model = None
optimized_model = None
scaler = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_models():
    """Load trained models from disk."""
    global baseline_model, optimized_model

    if tf is None:
        return

    try:
        if os.path.exists('models/baseline_cnn.h5'):
            baseline_model = tf.keras.models.load_model('models/baseline_cnn.h5')
    except Exception:
        baseline_model = None

    try:
        if os.path.exists('models/pso_optimized_cnn.h5'):
            optimized_model = tf.keras.models.load_model('models/pso_optimized_cnn.h5')
    except Exception:
        optimized_model = None


def preprocess_input(data_array):
    """Preprocess input data for prediction."""
    # Normalize (assuming standard scaler with mean=0, std=1)
    data_array = (data_array - np.mean(data_array)) / (np.std(data_array) + 1e-8)
    return np.expand_dims(data_array, axis=-1)


def rule_based_probability(features):
    """Fallback probability when trained models are unavailable."""
    ph, tds, water_level, dht_temp, dht_humidity, water_temp = [float(v) for v in features]

    checks = [
        5.5 <= ph <= 7.0,
        800 <= tds <= 1500,
        0 <= water_level <= 3,
        18 <= dht_temp <= 28,
        50 <= dht_humidity <= 90,
        16 <= water_temp <= 26,
    ]
    score = sum(checks) / len(checks)

    # Keep prediction away from extreme 0/1 for better UX confidence values
    return max(0.05, min(0.95, score))


def format_prediction(probability, source='model'):
    """Normalize output schema for frontend."""
    return {
        'probability': float(probability),
        'health_status': 'Healthy' if probability > 0.5 else 'Unhealthy',
        'confidence': float(max(probability, 1 - probability)),
        'source': source
    }


def fallback_predictions(features):
    """Generate baseline/optimized predictions without saved models."""
    prob = rule_based_probability(features)
    return {
        'baseline': format_prediction(prob, source='rule_based_fallback'),
        'optimized': format_prediction(prob, source='rule_based_fallback')
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    if os.path.exists('dashboard.html'):
        return render_template('dashboard.html')
    return jsonify({'error': 'dashboard.html not found'}), 404


@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tensorflow_available': tf is not None,
        'baseline_model_loaded': baseline_model is not None,
        'optimized_model_loaded': optimized_model is not None,
        'model_files_found': {
            'baseline': os.path.exists('models/baseline_cnn.h5'),
            'optimized': os.path.exists('models/pso_optimized_cnn.h5')
        }
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions using trained models.
    Expects JSON: {
        'ph': float,
        'tds': float,
        'water_level': int,
        'dht_temp': float,
        'dht_humidity': float,
        'water_temp': float
    }
    """
    try:
        data = request.json or {}

        # Extract features in correct order
        features = np.array([[
            data.get('ph', 6.0),
            data.get('tds', 1200),
            data.get('water_level', 1),
            data.get('dht_temp', 24),
            data.get('dht_humidity', 70),
            data.get('water_temp', 21)
        ]])

        # Preprocess
        features_processed = preprocess_input(features)

        predictions = {}

        # Baseline prediction
        if baseline_model is not None:
            baseline_pred = baseline_model.predict(features_processed, verbose=0)
            predictions['baseline'] = format_prediction(baseline_pred[0][0], source='baseline_model')

        # Optimized prediction
        if optimized_model is not None:
            optimized_pred = optimized_model.predict(features_processed, verbose=0)
            predictions['optimized'] = format_prediction(optimized_pred[0][0], source='optimized_model')

        # Fallback when no trained models are available
        if not predictions:
            predictions = fallback_predictions(features[0])

        return jsonify({
            'success': True,
            'input_data': data,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'used_fallback': baseline_model is None and optimized_model is None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']

        # Read CSV
        df = pd.read_csv(file)

        # Expected columns
        feature_cols = ['pH', 'TDS', 'water_level', 'DHT_temp', 'DHT_humidity', 'water_temp']

        if not all(col in df.columns for col in feature_cols):
            return jsonify({'success': False, 'error': 'Missing required columns'}), 400

        # Prepare features
        X_raw = df[feature_cols].values
        X = preprocess_input(X_raw)

        results = []

        if baseline_model is not None:
            baseline_preds = baseline_model.predict(X, verbose=0).flatten()
            results.append({
                'model': 'baseline',
                'source': 'baseline_model',
                'predictions': baseline_preds.tolist(),
                'labels': ['Healthy' if p > 0.5 else 'Unhealthy' for p in baseline_preds]
            })

        if optimized_model is not None:
            optimized_preds = optimized_model.predict(X, verbose=0).flatten()
            results.append({
                'model': 'optimized',
                'source': 'optimized_model',
                'predictions': optimized_preds.tolist(),
                'labels': ['Healthy' if p > 0.5 else 'Unhealthy' for p in optimized_preds]
            })

        if not results:
            fallback_probs = [rule_based_probability(row) for row in X_raw]
            fallback_labels = ['Healthy' if p > 0.5 else 'Unhealthy' for p in fallback_probs]
            results = [
                {
                    'model': 'baseline',
                    'source': 'rule_based_fallback',
                    'predictions': fallback_probs,
                    'labels': fallback_labels
                },
                {
                    'model': 'optimized',
                    'source': 'rule_based_fallback',
                    'predictions': fallback_probs,
                    'labels': fallback_labels
                }
            ]

        return jsonify({
            'success': True,
            'num_samples': len(df),
            'results': results,
            'used_fallback': baseline_model is None and optimized_model is None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/metrics')
def get_metrics():
    """Get model metrics from saved results."""
    try:
        if os.path.exists('results/metrics_comparison.json'):
            with open('results/metrics_comparison.json', 'r') as f:
                metrics = json.load(f)
            return jsonify({'success': True, 'metrics': metrics, 'source': 'saved_file'})

        fallback_metrics = {
            'baseline': {'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A', 'auc': 'N/A'},
            'optimized': {'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1_score': 'N/A', 'auc': 'N/A'},
            'note': 'No saved metrics file found. Run training to generate results/metrics_comparison.json.'
        }
        return jsonify({'success': True, 'metrics': fallback_metrics, 'source': 'fallback'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-report')
def generate_report():
    """Generate comprehensive report"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'project': 'Hybrid Metaheuristic Optimization of Deep Learning for Hydroponics',
            'models': {
                'baseline': 'CNN with default hyperparameters',
                'optimized': 'CNN with PSO-optimized hyperparameters'
            },
            'dataset': {
                'source': 'IoT Hydroponic Sensor Data',
                'total_samples': 50570,
                'features': ['pH', 'TDS', 'water_level', 'DHT_temp', 'DHT_humidity', 'water_temp'],
                'target': 'System Health (Binary Classification)'
            },
            'optimization_method': 'Particle Swarm Optimization (PSO)',
            'pso_config': {
                'particles': 5,
                'iterations': 10,
                'inertia_range': [0.4, 0.9],
                'cognitive_param': 2.0,
                'social_param': 2.0
            }
        }

        return jsonify({'success': True, 'report': report})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Load models once when app is imported (e.g., gunicorn/Render/Vercel backend).
load_models()

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == '__main__':
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

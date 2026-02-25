# Hybrid Metaheuristic Optimization of Deep Learning Models for Hydroponic Agriculture

## 🌱 Project Overview

This project implements a **CNN-PSO hybrid approach** for optimizing deep learning models in hydroponic agriculture systems. The system analyzes IoT sensor data (pH, TDS, temperature, humidity, water level) from hydroponic systems and predicts the overall system health using:

1. **Baseline CNN**: Standard CNN with default hyperparameters
2. **PSO-Optimized CNN**: CNN with hyperparameters optimized using Particle Swarm Optimization
3. **Comparison Analysis**: Comprehensive comparison and visualization of both approaches

### Key Features
- ✅ Complete data preprocessing pipeline for IoT sensor data
- ✅ Custom CNN architecture for 1D time-series data
- ✅ Particle Swarm Optimization for hyperparameter tuning
- ✅ Real-time prediction API with Flask web interface
- ✅ Comprehensive visualization and performance metrics
- ✅ Batch processing capability
- ✅ Production-ready deployment setup

---

## 📊 Dataset

**IoTData-Raw.csv** contains:
- **50,570 records** from hydroponic system sensors
- **6 numeric features**: pH, TDS, water_level, DHT_temp, DHT_humidity, water_temp
- **5 control features**: pH_reducer, add_water, nutrients_adder, humidifier, ex_fan
- **Target variable** (engineered): system_health (binary: 0=unhealthy, 1=healthy)

**Healthy Conditions:**
- pH: 5.5-7.0
- TDS: 800-1500 ppm
- Temperature: 18-28°C
- Humidity: 50-90%
- Water Temp: 16-26°C

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hydroponic-ml-optimization.git
cd hydroponic-ml-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Run complete training pipeline
python train.py

# This will:
# - Load and preprocess IoT data
# - Train baseline CNN model
# - Run PSO optimization
# - Train optimized CNN model
# - Generate 4 comprehensive visualization plots
# - Save models to 'models/' directory
```

**Expected Output:**
- `models/baseline_cnn.h5` - Baseline model
- `models/pso_optimized_cnn.h5` - Optimized model
- `results/01_training_history.png` - Training curves
- `results/02_metrics_comparison.png` - Performance comparison
- `results/03_confusion_matrices.png` - Confusion matrices
- `results/04_roc_curves.png` - ROC-AUC curves

### 3. Run Web Interface

```bash
# Start Flask server
python flask_app.py

# Open browser: http://localhost:5000
```

**Features:**
- 📊 Real-time system health prediction
- 📁 Batch CSV processing
- 📈 Model performance metrics
- 🏗️ System architecture visualization

### 4. Deploy Frontend + Backend (Automatic)

```bash
# From repository root (recommended)
./deploy/deploy_auto.sh
```

`deploy_auto.sh` will try Docker deployment first and automatically fall back to a local Python deployment if Docker is unavailable.

- Docker mode URL: `http://localhost:${FRONTEND_PORT:-80}`
- Local fallback URL (single service serving UI + API): `http://localhost:5000`

**Stop commands:**
```bash
# Stop Docker services
docker compose -f deploy/docker-compose.prod.yml down

# Stop local fallback service
./deploy/stop_local.sh
```

**Optional environment variables (Docker mode):**
- `FRONTEND_PORT` (default `80`)
- `BACKEND_IMAGE` (default `hydroponic-backend:latest`)
- `FRONTEND_IMAGE` (default `hydroponic-frontend:latest`)

### 4. API Usage (cURL)

```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 6.0,
    "tds": 1200,
    "water_level": 1,
    "dht_temp": 24,
    "dht_humidity": 70,
    "water_temp": 21
  }'

# Health check
curl http://localhost:5000/api/health

# Get metrics
curl http://localhost:5000/api/metrics
```

---

## 📁 Project Structure

```
hydroponic-ml-optimization/
├── IoTData-Raw.csv              # Your IoT dataset
├── train.py                      # Training pipeline with visualizations
├── flask_app.py                  # Web API server
├── config.py                     # Configuration management
├── core_modules.py               # ML modules (DataHandler, CNN, PSO, etc.)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── models/                       # Trained models
│   ├── baseline_cnn.h5
│   ├── pso_optimized_cnn.h5
│   └── hyperparameters.json
│
├── results/                      # Output visualizations
│   ├── 01_training_history.png
│   ├── 02_metrics_comparison.png
│   ├── 03_confusion_matrices.png
│   └── 04_roc_curves.png
│
├── templates/                    # HTML templates
│   └── index.html               # Web interface
│
├── static/                       # Static files (CSS, JS)
│   └── style.css
│
└── logs/                         # Training logs
```

---

## 🧠 Model Architecture

### Baseline CNN
```
Input (6 features)
    ↓
Conv1D(32, 3) → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(64, 3) → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(128, 3) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Flatten
    ↓
Dense(256) → ReLU → BatchNorm → Dropout(0.3)
Dense(128) → ReLU → BatchNorm → Dropout(0.3)
Dense(64) → ReLU → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Binary Output
```

### PSO Optimization

**Search Space:**
- Learning Rate: [0.0001, 0.01]
- Batch Size: [16, 64]
- Dropout Rate: [0.1, 0.5]
- Dense Units: [64-256, 32-128, 16-64]

**Configuration:**
- Particles: 5
- Iterations: 10
- Inertia Weight: [0.4, 0.9]
- Cognitive Coefficient: 2.0
- Social Coefficient: 2.0

---

## 📊 Expected Results

### Performance Comparison

| Metric | Baseline CNN | PSO-Optimized | Improvement |
|--------|-------------|---------------|-------------|
| Accuracy | ~90-92% | ~93-95% | +2-3% ↑ |
| Precision | ~89% | ~92% | +3% ↑ |
| Recall | ~91% | ~94% | +3% ↑ |
| F1-Score | ~90% | ~93% | +3% ↑ |
| AUC-ROC | ~0.95 | ~0.97 | +0.02 ↑ |

### Computational Efficiency

| Metric | Grid Search | PSO Approach |
|--------|------------|-------------|
| Training Epochs | 640 | 125 |
| Estimated Time | 8-10 hours | 1.5-2 hours |
| Speed Improvement | - | 75-80% faster |

---

## 🎯 Key Hyperparameters

### Training Configuration
```python
BASELINE_EPOCHS = 50
OPTIMIZED_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001  # Baseline
DROPOUT_RATE = 0.3     # Baseline
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
EARLY_STOPPING_PATIENCE = 10
```

### PSO Configuration
```python
PSO_PARTICLES = 5
PSO_ITERATIONS = 10
PSO_INERTIA_MIN = 0.4
PSO_INERTIA_MAX = 0.9
PSO_COGNITIVE = 2.0
PSO_SOCIAL = 2.0
PSO_TOPOLOGY = 'global'
```

---

## 📈 Visualizations Generated

### 1. Training History
- Accuracy curves (Train/Val) for both models
- Loss curves (Train/Val) for both models
- Precision curves
- Recall curves
- AUC curves

### 2. Metrics Comparison
- Side-by-side bar chart of all metrics
- Percentage improvement visualization
- Statistical comparison

### 3. Confusion Matrices
- Baseline CNN confusion matrix
- PSO-Optimized CNN confusion matrix
- True/False positives and negatives

### 4. ROC-AUC Curves
- Baseline CNN ROC curve
- PSO-Optimized CNN ROC curve
- AUC scores comparison

---

## 🌐 Web Interface Features

### Overview Tab
- Project information
- Dataset statistics
- Model architecture details
- Key metrics summary

### Real-Time Prediction Tab
- Interactive sensor input forms
- Live health status prediction
- Confidence scores
- Baseline vs Optimized comparison

### Batch Processing Tab
- CSV file upload
- Bulk predictions
- Results export

### Metrics Tab
- Model performance metrics
- Accuracy, Precision, Recall, F1, AUC
- Performance timeline

### Architecture Tab
- System architecture diagram
- CNN layer structure
- PSO optimization pipeline
- Data flow visualization

---

## 🔧 Advanced Usage

### Custom Training with Different Dataset
```python
from core_modules import DataHandler, Trainer, PSOOptimizer

# Load data
handler = DataHandler(config)
X, y, features = handler.load_and_preprocess('your_data.csv')

# Train baseline
trainer = Trainer(config)
model = trainer.train_model(X_train, y_train, X_val, y_val)

# Run PSO optimization
optimizer = PSOOptimizer(config)
best_params, history = optimizer.optimize(X_train, y_train, X_val, y_val)
```

### Deploy to Production
```bash
# Using Gunicorn for production
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app

# Using Docker (optional)
docker build -t hydroponic-ml .
docker run -p 5000:5000 hydroponic-ml
```

### Monitor Training Progress
```bash
# Using TensorBoard
tensorboard --logdir=logs/

# View at http://localhost:6006
```

---

## 🐛 Troubleshooting

### Issue: Memory Error During Training
**Solution:** Reduce BATCH_SIZE in config.py
```python
BATCH_SIZE = 16  # Instead of 32
```

### Issue: Poor Model Performance
**Solution:** Adjust PSO search space or increase iterations
```python
PSO_ITERATIONS = 15  # Increase from 10
PSO_PARTICLES = 7    # Increase from 5
```

### Issue: Models Not Loading in Web Interface
**Solution:** Ensure models are trained and saved
```bash
python train.py  # Run training first
```

---

## 📚 Referenced Research

1. **PSO-CNN-Bi-LSTM for Crop Yield Prediction**
   - Saini et al. (2024) - Environmental Modeling & Assessment
   - MAE: 0.39 tonnes/hectare

2. **CNN Architecture Search using PSO**
   - Wang et al. (2020) - Journal of Artificial Intelligence Research
   - 95+ citations

3. **Hydroponic Disease Detection**
   - CNN-based system with 86-99% accuracy
   - IoT sensor integration

4. **Metaheuristic Optimization in Agriculture**
   - Multiple optimizer comparisons
   - Whale Optimization, Genetic Algorithms, PSO

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@project{HydroponicMLOptimization2024,
  title={Hybrid Metaheuristic Optimization of Deep Learning Models for Hydroponic Agriculture},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hydroponic-ml-optimization}
}
```

---

## 📧 Support & Questions

For questions or issues:
1. Check Troubleshooting section
2. Review GitHub issues
3. Contact: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- PySwarms for PSO implementation
- Scikit-learn for machine learning utilities
- Flask for web framework
- Contributors and researchers in hydroponic agriculture ML

---

**Last Updated:** January 25, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready

---

## 🚢 Automated Deployment (Frontend + Backend)

This repository now includes a fully automated deployment pipeline:

- **Backend container**: `Dockerfile.backend` (Flask API served with Gunicorn)
- **Frontend container**: `Dockerfile.frontend` (Nginx serving `index.html` and proxying `/api` to backend)
- **Production compose**: `deploy/docker-compose.prod.yml`
- **GitHub Actions CI/CD**: `.github/workflows/deploy.yml`

### 1) One-time server prerequisites

On your deployment server:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable --now docker
mkdir -p /opt/hydroponic-pso
```

### 2) Configure GitHub repository secrets

Set the following **Repository Secrets**:

- `SSH_HOST` - server host or IP
- `SSH_USER` - SSH username
- `SSH_PRIVATE_KEY` - private key for SSH auth
- `DEPLOY_PATH` - remote path (e.g., `/opt/hydroponic-pso`)
- `GHCR_USER` - GitHub username for GHCR pull
- `GHCR_PAT` - GitHub Personal Access Token with `read:packages`

### 3) Trigger deployment

- Push to the `main` branch, or
- Run **Actions → Build and Deploy Frontend + Backend → Run workflow**

The workflow will:
1. Build backend and frontend Docker images.
2. Push them to GHCR.
3. SSH into your server.
4. Pull latest images and restart both services with Docker Compose.

### 4) Access your deployed app

- Frontend: `http://<your-server-ip>/`
- Backend health endpoint: `http://<your-server-ip>/api/health`

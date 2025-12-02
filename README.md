📡 Telecom Churn Prediction System

📖 Project Overview

This is an end-to-end Machine Learning solution designed to predict customer churn in the telecommunications industry. The project goes beyond simple modeling by implementing a production-ready pipeline that includes robust data preprocessing, handling of class imbalance, model explainability, and a deployable REST API with a user-friendly frontend.

The goal is to identify customers at risk of leaving (churning) so businesses can take proactive retention actions.

🌟 Key Features

Competition-Grade ML Pipeline: Utilizes XGBoost with Stratified K-Fold Cross-Validation for robust performance.

Imbalance Handling: Implements SMOTE and class weighting to accurately detect the minority churn class.

Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) to provide transparent reasons behind every prediction.

Production API: High-performance REST API built with FastAPI.

Interactive UI: Responsive HTML/CSS/JS frontend for real-time demonstrations.

Deployment Ready: Includes Docker support and is configured for cloud platforms like Render.

🛠️ Tech Stack

Core: Python 3.11

Machine Learning: scikit-learn, xgboost, imbalanced-learn (SMOTE)

Data Processing: pandas, numpy

Explainability: shap

API Framework: FastAPI, Uvicorn

Frontend: HTML5, CSS3, Vanilla JavaScript

DevOps: Docker, Render (Cloud Hosting)

📂 Project Structure

telecom-churn-project/
├── data/
│   └── telco_churn.csv       # Dataset (Input)
├── models/
│   └── churn_xgb.joblib      # Trained Model Artifacts
├── notebooks/
│   └── exploration.ipynb     # EDA and experimentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning & transformation pipelines
│   ├── features.py           # Feature engineering logic
│   ├── train.py              # Training script (CV, Hyperparams, Saving)
│   ├── model.py              # Inference class (Model loading & prediction)
│   ├── explain.py            # SHAP explanation utilities
│   └── api_main.py           # FastAPI entry point
├── web/
│   ├── index.html            # User Interface
│   └── styles.css            # Styling
├── Dockerfile                # Container configuration
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation


🚀 Getting Started

Prerequisites

Python 3.9 or higher

Git

1. Clone the Repository

git clone [https://github.com/yourusername/telecom-churn-project.git](https://github.com/yourusername/telecom-churn-project.git)
cd telecom-churn-project


2. Install Dependencies

It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Train the Model

Run the training pipeline to generate the model artifact (churn_xgb.joblib).

python src/train.py --data data/telco_churn.csv --out models/churn_xgb.joblib


This script applies preprocessing, SMOTE, and trains an XGBoost ensemble using 5-fold CV.

4. Run the API (Backend)

Start the FastAPI server locally.

uvicorn src.api_main:app --host 0.0.0.0 --port 8000 --reload


The API documentation will be available at http://localhost:8000/docs.

5. Run the Frontend (UI)

Simply open web/index.html in your browser.
Note: Ensure the backend is running before testing predictions.

🌐 Deployment (Render Guide)

This project is configured for seamless deployment on Render.

Backend (Web Service)

Link your repo to Render.

Select Web Service with Python 3 runtime.

Build Command: pip install -r requirements.txt

Start Command: uvicorn src.api_main:app --host 0.0.0.0 --port $PORT

Frontend (Static Site)

Create a Static Site on Render linked to the same repo.

Publish Directory: web

Update the fetch URL in web/index.html to point to your new Backend URL.

📊 Performance & Metrics

The model focuses on maximizing Average Precision (AP) and Recall to capture as many potential churners as possible without overwhelming the retention team with false alarms.

ROC-AUC: ~0.85 (Validation Avg)

Precision-Recall AUC: ~0.68

Recall (Churn Class): ~0.80

🔮 Future Improvements

Add Optuna for automated hyperparameter tuning.

Implement LightGBM and CatBoost for model comparison.

Add a Drift Detection module to monitor data changes over time.

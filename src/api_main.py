from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from src.model import EnsembleModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Telecom Churn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'models/churn_xgb.joblib'
model = EnsembleModel.load(MODEL_PATH)

# Load the training feature columns to ensure alignment
FEATURE_COLUMNS_PATH = 'models/feature_columns.joblib'
try:
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
except:
    print("Warning: feature_columns.joblib not found. Feature alignment may fail.")
    feature_columns = None

class Customer(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    PhoneService: str = "Yes"
    InternetService: str = "Fiber optic"
    Contract: str

def feature_engineering(df):
    """Apply same feature engineering as training"""
    df = df.copy()
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72,100], labels=False)
    return df

def prepare_features(df):
    """Prepare features matching training pipeline"""

    df = feature_engineering(df)
    
    yes_no_map = {'Yes': 1, 'No': 0}
    for col in ['Partner', 'Dependents', 'PhoneService']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map(yes_no_map)
    
   
    df = pd.get_dummies(df, drop_first=True)
    
  
    if feature_columns is not None:
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_columns]
    
    
    df = df.fillna(0)
    
    return df

@app.post('/predict')
def predict(item: Customer):
    try:
        
        d = item.dict()
        print(f"Received data: {d}")
        df = pd.DataFrame([d])
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
  
        df_processed = prepare_features(df)
        print(f"Processed shape: {df_processed.shape}")
        print(f"Processed columns: {df_processed.columns.tolist()}")
        
      
        proba = model.predict_proba(df_processed)[0]
        print(f"Prediction: {proba}")
        
        return {
            "churn_probability": float(proba),
            "churn": bool(proba >= 0.5),
            "risk_level": "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
        }
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get('/health')
def health():
    return {"status": "healthy", "model_loaded": model is not None}
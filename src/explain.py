import shap
import joblib
import pandas as pd
from model import EnsembleModel

def explain_sample(model_path, X_sample):
    models = joblib.load(model_path)
    m = models[0]

    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample)

    return shap_values

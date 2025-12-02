import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocessing import load_data, basic_cleaning

def feature_engineering(df):
    df = df.copy()
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72,100], labels=False)
    return df

def prepare_Xy(df, target='Churn'):
    # Handle target
    if target in df.columns:
        if df[target].dtype == 'object':
            y = df[target].map({'Yes': 1, 'No': 0})
        else:
            y = df[target].astype(int)
    else:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])

   
    X = pd.get_dummies(X, drop_first=True)

  
    X = X.fillna(0)

    return X, y


def train_cv(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof = np.zeros(len(y))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # Apply SMOTE
        sm = SMOTE(random_state=seed)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric='logloss'
        )

        model.fit(
            X_tr_res, 
            y_tr_res,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_val)[:,1]
        oof[val_idx] = preds

        models.append(model)
        auc = roc_auc_score(y_val, preds)
        ap = average_precision_score(y_val, preds)
        print(f"Fold {fold+1} - AUC: {auc:.4f}, AP: {ap:.4f}")

    overall_auc = roc_auc_score(y, oof)
    overall_ap = average_precision_score(y, oof)
    print(f"\nOverall - AUC: {overall_auc:.4f}, AP: {overall_ap:.4f}")
    
    return models, oof

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training CSV file')
    parser.add_argument('--out', default='models/churn_xgb.joblib', help='Output model path')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for Churn column
    if 'Churn' not in df.columns:
        print("ERROR: 'Churn' column not found!")
        print("Available columns:", df.columns.tolist())
        exit(1)
    
    print("\nCleaning data...")
    df = basic_cleaning(df)
    
    print("Applying feature engineering...")
    df = feature_engineering(df)

    print("\nObject columns after processing:")
    print(df.select_dtypes(include='object').columns.tolist())
    
    print("\nPreparing features...")
    X, y = prepare_Xy(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")

    print("\nTraining models with cross-validation...")
    models, oof = train_cv(X, y)

 
    joblib.dump(models, args.out)
    print(f"\n✓ Saved {len(models)} models to {args.out}")
    
    
    feature_cols_path1 = args.out.replace('.joblib', '_features.joblib')
    feature_cols_path2 = 'models/feature_columns.joblib'
    
    joblib.dump(X.columns.tolist(), feature_cols_path1)
    joblib.dump(X.columns.tolist(), feature_cols_path2)
    print(f"✓ Saved feature columns to {feature_cols_path1}")
    print(f"✓ Saved feature columns to {feature_cols_path2}")
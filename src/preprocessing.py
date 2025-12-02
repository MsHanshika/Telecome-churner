import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CATEGORICALS = None
NUMERICALS = None

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    df = df.copy()

    # Remove whitespace & unify missing categories
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('', 'Unknown')
        df[col] = df[col].fillna('Unknown')

    # Convert Yes/No to 1/0
    yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes':1, 'No':0})

    return df



def get_feature_sets(df):
    global CATEGORICALS, NUMERICALS
    CATEGORICALS = df.select_dtypes(include=['object']).columns.tolist()
    NUMERICALS = df.select_dtypes(include=[np.number]).columns.tolist()
    return CATEGORICALS, NUMERICALS

class TabularTransformer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_fill = {}

    def fit_transform(self, X):
        X = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        self.scaler.fit(X_num)
        X_num_scaled = pd.DataFrame(self.scaler.transform(X_num), columns=X_num.columns, index=X.index)

        X_cat = pd.get_dummies(X.select_dtypes(include=['object']), drop_first=True)

        self._cat_columns = X_cat.columns

        return pd.concat([X_num_scaled, X_cat], axis=1)

    def transform(self, X):
        X_num = X.select_dtypes(include=[np.number])
        X_num_scaled = pd.DataFrame(self.scaler.transform(X_num), columns=X_num.columns, index=X.index)

        X_cat = pd.get_dummies(X.select_dtypes(include=['object']), drop_first=True)
        X_cat = X_cat.reindex(columns=self._cat_columns, fill_value=0)

        return pd.concat([X_num_scaled, X_cat], axis=1)

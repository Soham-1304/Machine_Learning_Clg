import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

def quick_summary(df: pd.DataFrame) -> None:
    """
    Print a fast profiling summary of a DataFrame:
    shape, dtypes, missing values, and basic stats.
    Reusable version of the manual EDA steps from Assignment 2.
    """
    print(f"Shape: {df.shape}")
    print("\nData Types:\n",df.dtypes)
    missing = df.isnull().sum()
    missing = missing[missing>0]
    print("\nMissing Values:\n",missing if not missing.empty else "None")
    print("\nBasic Stats:\n",df.describe())

def split_data(df:pd.DataFrame, target_column:str, test_size:float=0.2, random_state:int=42):
    """
    Split a DataFrame into train/test features and labels.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X,y,test_size=test_size,random_state=random_state)

def scale_features(X_train:pd.DataFrame, X_test:pd.DataFrame):
    """
    Scale numeric features using StandardScaler.
    Fit on training data only, then apply to both train and test
    (prevents data leakage from test set into training).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled,X_test_scaled,scaler

def evaluate_model(y_true, y_pred, task:str="classification")->dict:
    """
    Compute basic evaluation metrics.
    task = 'classification' -> accuracy
    task = 'regression' -> MSE and R-squared
    """
    if task == "classification":
        return {"accuracy":accuracy_score(y_true,y_pred)}
    elif task == "regression":
        return {
            "mse": mean_squared_error(y_true,y_pred),
            "r2" : r2_score(y_true,y_pred)
        }
    else:
        raise ValueError("task must be 'classification' or 'regression'")


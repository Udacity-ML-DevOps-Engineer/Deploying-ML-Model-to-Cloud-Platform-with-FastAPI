import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import fbeta_score, precision_score, recall_score
from .data import process_data
import joblib
import logging
import json


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta



def compute_slice_metrics(
    data: pd.DataFrame,
    feature: str,
    model,
    encoder,
    lb,
    cat_features: list
) -> pd.DataFrame:
    """
    Computes performance metrics on slices of data for a given feature
    
    Args:
        data: pandas dataframe
        feature: feature to slice on
        model: trained model
        encoder: trained encoder
        lb: trained label binarizer
        cat_features: list of categorical features
    
    Returns:
        pd.DataFrame: DataFrame containing the performance metrics for each slice
    """
    slice_metrics = []
    
    # Get unique values in the feature
    unique_values = data[feature].unique()
    
    for value in unique_values:
        # Get slice of data where feature equals the current value
        slice_data = data[data[feature] == value].copy()
        
        if len(slice_data) < 1:
            continue
            
        # Process the slice
        X_test, y_test, _, _ = process_data(
            slice_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        # Get predictions
        preds = inference(model, X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, preds, zero_division=1)
        recall = recall_score(y_test, preds, zero_division=1)
        fbeta = fbeta_score(y_test, preds, beta=1, zero_division=1)
        
        # Store metrics
        slice_metrics.append({
            "slice": str(value),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "number_of_samples": len(slice_data)
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(slice_metrics)
    
    return metrics_df


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


import sys
import os
import pytest
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.ml.data import process_data
from train.ml.model import train_model, compute_model_metrics, inference

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.ml.data import process_data
from train.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """
    Fixture for loading test data
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "census_cleaned.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    return pd.read_csv(data_path)

def test_process_data(data):
    """
    Test process_data function
    """
    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]
    
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    assert X.shape[0] == data.shape[0]
    assert len(y) == data.shape[0]
    assert encoder is not None
    assert lb is not None

def test_train_model(data):
    """
    Test train_model function
    """
    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]
    
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    model = train_model(X, y)
    assert model is not None

def test_inference(data):
    """
    Test inference function
    """
    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]
    
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    model = train_model(X, y)
    preds = inference(model, X)
    
    assert len(preds) == len(y)
    assert all(isinstance(pred, (int, bool, np.integer)) for pred in preds)

def test_compute_model_metrics():
    """
    Test compute_model_metrics function
    """
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


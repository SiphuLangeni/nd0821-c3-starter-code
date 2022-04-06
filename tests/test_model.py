import numpy as np
import pandas as pd
import pytest
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data import process_data
from src.model import compute_model_metrics, inference, train_model


@pytest.fixture
def y():
    return np.array([1, 1, 1, 1])

@pytest.fixture
def preds():
    return np.array([0, 0, 0, 1])

@pytest.fixture
def model():
    return load('model/rfc_model.joblib')

def X():
    return np.array([1, 1, 1, 1])


@pytest.fixture
def X_train():
    X_train = np.array([[3.30000e+01, 1.98183e+05, 1.30000e+01, 0.00000e+00, 0.00000e+00,
        5.00000e+01, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
        1.00000e+00, 0.00000e+00, 0.00000e+00]])

    return X_train

@pytest.fixture
def y_train():
    return np.array([1])

def test_compute_model_metrics(y, preds):
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert precision == 1.0
    assert recall == 0.25
    assert fbeta == 0.4


def test_inference(model, X_train):

    preds = inference(model, X_train)

    assert preds is not None



def test_train_model(X_train, y_train):
    rfc = RandomForestClassifier()
    model = rfc.fit(X_train, y_train)

    assert len(X_train) == len(y_train)

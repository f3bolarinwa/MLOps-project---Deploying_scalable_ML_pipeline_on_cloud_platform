"""
Test suite containing unit tests for custom functions and api.

Author: Femi Bolarinwa
Data: July 2023
"""

# importing required python libraries
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from fastapi.testclient import TestClient

# Import api app from main.py
from main import app

# importing custom functions
from ML.data import process_data
from ML.model import train_model, compute_model_metrics, inference

# Reading configuration file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


@pytest.fixture
def data():
    """
    pytest fixture to import data

    Input:

    Returns:
    df: pd.DataFrame
        input data
    """
    df = pd.read_csv('./data/census_clean.csv')

    return df


@pytest.fixture
def data_train(data):
    """
    pytest fixture for train-test split

    Input:
    df: pd.DataFrame
        input data

    Returns:
    X_train: pd.DataFrame
        training data (predictors)
    y_train: pd.DataFrame
        training data (target variable)
    X_test: pd.DataFrame
        test data (predictors)
    y_test: pd.DataFrame
        test data (target variable)
    """

    train, test = train_test_split(
        data, test_size=config['model_training']['test_size'],
        random_state=config['model_training']['random_state'], stratify=data['salary']
    )

    # categorical features in data
    cat_features = config['data']['cat_features']

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def train_model_for_test(data_train):
    """
    pytest fixture to train a machine learning model and return it.

    Inputs
    ------
    X_train: pd.DataFrame
        training data (predictors)
    y_train: pd.DataFrame
        training data (target variable)
    X_test: pd.DataFrame
        test data (predictors)
    y_test: pd.DataFrame
        test data (target variable)

    Returns
    -------
    model
        Trained machine learning model.
    """

    X_train, y_train, X_test, y_test = data_train

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    return classifier


def test_train_model(data_train):
    """
    valid test if train_model returns a model
    """

    X_train, y_train, X_test, y_test = data_train
    train_model(X_train, y_train)

    assert os.path.isfile("./model/classifier_model.joblib")


def test_compute_model_metrics(data_train, train_model_for_test):
    """
    valid test if compute_model_metrics returns all 3 metrics
    """

    X_train, y_train, X_test, y_test = data_train
    classifier = train_model_for_test
    y = y_train
    preds = classifier.predict(X_train)

    precision, recall, fbeta, accuracy = compute_model_metrics(y, preds)

    assert isinstance(fbeta, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(accuracy, float)


def test_inference(data_train, train_model_for_test):
    """
    valid test if inference returns prediction
    """

    X_train, y_train, X_test, y_test = data_train

    X = X_train
    model = train_model_for_test

    preds = inference(model, X)

    assert preds.shape[0] > 0


# Instantiating fastAPI testclient with our api app.
client = TestClient(app)


def test_api_get():
    """
    valid test if client GET returns expected greeting.
    """

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == 'Welcome! Please use the POST method to run ML inference on the API. Thank you!' # {"greeting": "Welcome!"}


def test_api_post():
    """
    valid test if client POST returns expected prediction.
    """
    request = client.post(
        "/inference",
        json={
            'age': 33,
            'workclass': 'Private',
            'fnlgt': 184,
            'education': 'HS-grad',
            'marital_status': 'Never-married',
            'occupation': 'Prof-specialty',
            'relationship': 'Not-in-family',
            'race': 'Black',
            'sex': 'Male',
            'hoursPerWeek': 60,
            'nativeCountry': 'United-States'})
    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}



def test_api_post_2():
    """
    valid test if client POST returns expected prediction.
    """
    request = client.post(
        "/inference",
        json={
            'age': 72,
            'workclass': 'Self-emp-inc',
            'fnlgt': 473748,
            'education': 'Masters',
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'hoursPerWeek': 50,
            'nativeCountry': 'United-States'})
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}


'''

json={'age': 33,
                                     'workclass': 'Private',
                                     'fnlgt': 149184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 60,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}

'''
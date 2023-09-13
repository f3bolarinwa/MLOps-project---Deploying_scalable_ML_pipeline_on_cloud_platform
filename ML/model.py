"""
The file contains a library of functions to train model, compute model metrics, 
make inference and compute model performance across slices of data

Author: Femi Bolarinwa
Data: July 2023
"""


#importing required python libraries
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd
import numpy as np

#importing custom made module
from ML.data import process_data


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

    classifier = XGBClassifier(random_state=42, n_estimators = 120, learning_rate = 0.1, max_depth = 10)
    #classifier = DecisionTreeClassifier(random_state=42)
    #classifier = RandomForestTreeClassifier(random_state=42, n_estimators=200)
    classifier.fit(X_train, y_train)

    return classifier


def compute_model_metrics(y, preds):
    """
    access the trained machine learning model performance using precision, recall, and F1-score.

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
    accuracy = accuracy_score(y, preds)#, zero_division=1)

    return precision, recall, fbeta, accuracy


def inference(model, X):
    """ Runs model inferences and returns the predictions.

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




def slice_data(df, model, cat_features, encoder, lb):
    """
    computes model performance across slices of data
    
    Inputs
    ------
    df: pd.Dataframe
        Dataframe containing the features and label. 
    model:
        machine learning model
    cat_features:
        categorical features in df
    encoder:sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, obtained from trained model.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, obtained from trained model.

    Returns
    -------
    metrics: np.array
        array of metrics (precision, recall, f1-score) across slices of df.
    """

    metrics = []

    for feature in cat_features:
        for cls in df[feature].unique():
            X, y, encoder, lb = process_data(df[df[feature]==cls],
                categorical_features=cat_features, 
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
                )

            preds = inference(model, X)
            precision, recall, fbeta, accuracy = compute_model_metrics(y, preds)
            line = "[%s %s] Precision: %s " "Recall: %s FBeta: %s" % (feature, cls, precision, recall, fbeta)
            metrics.append(line)

    metrics = np.array(metrics)
    
    return metrics

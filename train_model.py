"""
Script to train machine learning model and access perfromance.

Author: Femi Bolarinwa
Data: July 2023
"""

# importing required python libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import yaml

# importing custom made modules
from ML.model import train_model, compute_model_metrics, inference, slice_data, process_data


# Reading configuration file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# loading in the data.
df = pd.read_csv('./data/census_clean.csv')

# train-test split
train, test = train_test_split(df, test_size=config['model_training']['test_size'],
                               random_state=config['model_training']['random_state'],
                               stratify=df['salary'])

# categorical features in data
cat_features = config['data']['cat_features']

# process training data
X_train, y_train, encoder_train, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# save encoder and binarizer of the training data
joblib.dump(encoder_train, './model/encoder_train.joblib')
joblib.dump(lb_train, './model/lb_train.joblib')


# process test data
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder_train, lb=lb_train
)

# Train and save a model.
classifier = train_model(X_train, y_train)
joblib.dump(classifier, './model/classifier_model.joblib')

# make prediction on test test and compute performance metrics
preds = inference(classifier, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print('precision on test set = {}'.format(precision))
print('recall on test set = {}'.format(recall))
print('fbeta on test set = {}'.format(fbeta))

# Model performance on slices of test data
model = classifier
metrics = slice_data(test, model, cat_features, encoder_train, lb_train)

# writing performances to a text file
with open('./ML/slice_output.txt', 'w') as out:
    for metric in metrics:
        out.write(metric + '\n')

# Model Card
Author: Femi Bolarinwa
Last updated: July 2023

## Model Details
This is a random forest classifier model to classify the income category of individuals based on various demographic features. The model is trained on publicly available census bureau data available here: https://archive.ics.uci.edu/dataset/20/census+income. In this model card, you can review quantitative components of the models performance and data, and also information about models intended uses, limitations and ethical consideration. 

## Intended Use
This model is intended to be used to predict the income category of individuals based on specific demographic information.

## Training Data
The training data includes 70% of the original dataset with the distribution of the target variable (salary) maintained from the original data set

## Evaluation Data
The test/evaluation data includes 30% of the original dataset with the distribution of the target variable (salary) maintained from the original data set

## Metrics
The model was evaluated on the test/evaluation data using precison, recall, f1-score and accuracy. The scores are:

precision = 0.70
recall = 0.60
f1-score = 0.64
accuracy = 0.84

## Ethical Considerations
Because the census data are collected directly from individuals, the data is reflective of how people perceive themselves and is likely to contain significant bias. People's perception can be influence by politics, nationality, geography, etc


## Caveats and Recommendations
Before using the machine learning model developed in this project, note that basic methods and parameters were use do build the model because the main focus of the project on MLOps skills like CI/CD and model deployment
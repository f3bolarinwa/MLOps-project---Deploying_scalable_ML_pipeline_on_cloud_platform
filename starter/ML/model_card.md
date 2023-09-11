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
The model was evaluated on the test/evaluation data using precison, recall and f1-score. The scores are:

precision = 0.67
recall = 0.56
f1-score = 0.61

## Ethical Considerations
Census data are data collected from people. The answers depend on how those people would like themselves to be seen and represented. And depend on how much people want to reveal about themselves. This kind od data can be influenced by politics, other people from the bubbles we live in and our wishes. As per slice performance metrics, some categorical values bring ideal predition. The data are insufficient for 'nationality' feature to trust that the model would handle well larger data set.

Because the census data are collected directly from individuals, the data is reflective of how people perceive themselves and is likely to contain significant bias. People's perception can be influence by politics, nationality, geography, etc


## Caveats and Recommendations
This model has very basic parameters set and before it is used to do real job, the parameters scope should be extended. As this project was about CI/CD, more emphasis was put on having the project utilizes tools like GitHub, dvc+S3 and Heroku than on how well to configure the model

Before using the machine learning model developed in this project, note that basic methods and parameters were use do build the model because the main focus of the project on MLOps skills like CI/CD and model deployment
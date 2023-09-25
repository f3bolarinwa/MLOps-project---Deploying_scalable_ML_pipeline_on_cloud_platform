# Project: Deploying a ML Model to Cloud Application Platform with FastAPI

A program requirement for Machine Learning DevOps Engineer Nanodegree @ Udacity School of Artificial Intelligence

## Project Description

In this project, I developed a classification model on publicly available Census Bureau data to predict salary range. I created and automated unit tests to monitor the model performance on various data slices. Then, deployed the ML model on a cloud platform (Render) using a REST API framework (FastAPI) and created API tests. The data slice validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

## Repository Content Description

Overview of the content in repository:

1)config.yaml: configuration file containing parameters for ML model training. 

2)environment.yml: contains all package dependencies for conda environment isolation for local development

3)main.py: contains and drives API

4)model_card.md: contains ML model details and summary

5)requirements.txt: contains all package requirements for CI/CD with github actions and environment building on cloud platform

6)sanitycheck.py: runs sanity check on API

7)setup.py: contains setup tools

8)test_suite.py: contains unit tests for ML functions and API

9)train_model.py: script to build, save and access ML model

10)Data: contains original data, cleaned version and jupyter notebook used for data cleaning

11)ML: contains custom functions to process data and build ML model

12)Model: contains ML model, its encoder and binarizer saved in joblib format

13).github/workflows: contains Github Actions workflow for continuous integration


## Running Files

1)Clone git repository to local machine

2)To isolate runtime environment on local machine, run in terminal:

> conda env create -f environment.yml

> conda activate proj3

3)install required versions of pip packages

> pip install -r requirements.txt  

4)Familiarize with API by perusing API front end live on render cloud platform:

https://femis-ml-pipeline-app.onrender.com/docs

5)To activate API locally, run:

> python main.py 

6)To test API locally, run:

> python run_request_on_local_ml_api.py

7)To test API live on cloud platform, run:

> python run_request_on_live_ml_api.py

8)Modify test files (run_request_on_local_ml_api.py, run_request_on_live_ml_api.py) to explore further

## API live on Render.com Cloud Platform

https://femis-ml-pipeline-app.onrender.com/docs

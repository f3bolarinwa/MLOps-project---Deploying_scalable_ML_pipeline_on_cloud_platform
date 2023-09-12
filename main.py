"""
Contains API to run inference on machine learning model

Author: Femi Bolarinwa
Data: July 2023
"""

# importing needed python libraries
import yaml
import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# importing custom made modules
from ML.schema import ModelInput
from ML.data import process_data

# Reading configuration file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# instantiating fastAPI
app = FastAPI()

# setting welcome message with fastAPI GET
@app.get("/")
async def welcome_message():
    return "Welcome! Please use the POST method to run ML inference on the API. Thank you!"

# making ML inference with fastAPI POST
@app.post("/inference")
async def run_inference(input_data: ModelInput):

    input_data = input_data.dict()
    change_keys = config['infer']['update_keys']
    columns = config['infer']['columns']
    cat_features = config['data']['cat_features']

    for new_key, old_key in change_keys:
        input_data[new_key] = input_data.pop(old_key)

    input_df = pd.DataFrame(
        data=input_data.values(),
        index=input_data.keys()).T
    input_df = input_df[columns]
    print(input_df)

    model = joblib.load("model/classifier_model.joblib")
    encoder = joblib.load("model/encoder_train.joblib")
    lb = joblib.load("model/lb_train.joblib")

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)

    pred = model.predict(X)
    prediction = lb.inverse_transform(pred)[0]

    return {"prediction": prediction}


if __name__ == "__main__":
    # uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

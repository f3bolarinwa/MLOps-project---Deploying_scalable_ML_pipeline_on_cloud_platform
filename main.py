"""
Contains API to run inference on machine learning model

Author: Femi Bolarinwa
Data: July 2023
"""

# importing needed python libraries
import os
import yaml
import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib

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
async def get_items():
    return {"greeting": "Welcome!"}

# making inference with fastAPI POST

try:

    @app.post("/inference")
    async def inference(input_data: ModelInput):

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

except PydanticUserError as exc_info:
    assert exc_info.code == 'model-field-overridden'

if __name__ == "__main__":
    # uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

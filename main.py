import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# -----------------------
# Load saved encoder and model
# -----------------------

# TODO: enter the path for the saved encoder
encoder_path = os.path.join("model", "encoder.pkl")
encoder = load_model(encoder_path)

# TODO: enter the path for the saved model
model_path = os.path.join("model", "model.pkl")
model = load_model(model_path)


# -----------------------
# Create RESTful API
# -----------------------
app = FastAPI()


# TODO: create a GET on the root giving a welcome message
@app.get("/")
def root():
    return {"message": "Hello from the API!"}


# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict
    data_dict = data.dict()
    
    # DO NOT MODIFY: convert dict to DataFrame, handle hyphen names
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    # List of categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process data using training = False (we only transform, do not fit)
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,       # no label for inference
        training=False,
        encoder=encoder
    )
    # Make prediction
    _inference = inference(model, data_processed)
    
    # Return prediction with labels applied
    return {"result": apply_label(_inference)}

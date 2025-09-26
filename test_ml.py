import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# Load some sample data for testing
data = pd.read_csv("./data/census.csv")
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

# Split data for testing
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# Process the data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # Test that the train_model function returns a RandomForestClassifier
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # Test that the inference function returns a numpy array of correct length
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # Test that compute_model_metrics returns 3 floats between 0 and 1
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    for val in [precision, recall, fbeta]:
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

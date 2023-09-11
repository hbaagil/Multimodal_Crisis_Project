import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crisis_helper.ml_logic.preprocess import text_cleaning
from crisis_helper.ml_logic.registry import load_model, load_vectorizer
from crisis_helper.ml_logic.model_binary import predict_binary_logistic_regression


app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#
@app.get("/predict")
def predict(twit: str):
    """
    Make a single course prediction.
    Assumes `twit` is provided as a string by the user
    """

    X_pred = pd.DataFrame([twit], columns=["tweet_text"])

    # Load vectorizer and model
    vectorizer = load_vectorizer()
    model = load_model()

    # Clean twits
    X_pred['clean_texts'] = X_pred.tweet_text.apply(text_cleaning)
    X_pred_unvec = X_pred['clean_texts']

    # Vectorize twits
    X_pred_vec = vectorizer.transform(X_pred_unvec)

    # Predict
    y_pred = predict_binary_logistic_regression(model, X_pred_vec)

    return {'twit_class': y_pred}


@app.get("/")
def root():
    return {'greeting': 'Hello'}

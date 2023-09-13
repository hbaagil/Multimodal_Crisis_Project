import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from crisis_helper.ml_logic.preprocess import text_cleaning
from crisis_helper.ml_logic.registry import load_model, load_vectorizer
from crisis_helper.ml_logic.registry import load_model_multiclass, load_vectorizer, load_img_model
from crisis_helper.ml_logic.model_binary import predict_binary_logistic_regression
from crisis_helper.ml_logic.model_multiclass import predict_multiclass_logistic_regression

import tensorflow as tf
from skimage.transform import resize
import cv2


app = FastAPI()
app.state.vectorizer = load_vectorizer()
app.state.binary_model = load_model()
app.state.multiclass_model = load_model_multiclass()
app.state.img_model = load_img_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict_binary")
def predict_binary(tweet: str):

    """
    Make a single course prediction.
    Assumes `tweet` is provided as a string by the user
    """

    X_pred = pd.DataFrame([tweet], columns=["tweet_text"])

    # Clean tweets
    X_pred['clean_texts'] = X_pred.tweet_text.apply(text_cleaning)
    X_pred_unvec = X_pred['clean_texts']

    # Vectorize tweets
    X_pred_vec = app.state.vectorizer.transform(X_pred_unvec)

    # Predict
    y_pred = predict_binary_logistic_regression(app.state.binary_model,
                                                X_pred_vec)

    return {'tweet_class': y_pred}

@app.get("/predict_multi")
def predict_multiclass(tweet: str):
    """
    Make a single course prediction.
    Assumes `tweet` is provided as a string by the user
    """

    X_pred = pd.DataFrame([tweet], columns=["tweet_text"])

    # Clean tweets
    X_pred['clean_texts'] = X_pred.tweet_text.apply(text_cleaning)
    X_pred_unvec = X_pred['clean_texts']

    # Vectorize tweets
    X_pred_vec = app.state.vectorizer.transform(X_pred_unvec)

    # Predict
    y_pred = predict_multiclass_logistic_regression(app.state.multiclass_model,
                                                    X_pred_vec)

    return {'tweet_class': y_pred}


@app.post('/upload_image')
async def predict_img(img: UploadFile=File(...)):

    ### Receiving and decoding the image
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    # Resize the image

    # Resize the image to (256, 256)
    resized_image = tf.image.resize(cv2_img, (256, 256))

    # Add an extra dimension for batch size
    reshaped_image = tf.expand_dims(resized_image, axis=0)

    # Classify
    y_pred = app.state.img_model.predict(reshaped_image)
    y_pred = y_pred.argmax(axis=1)

     # Define the label mapping
    label_mapping = {
        0: "affected_individuals",
        1: "infrastructure_and_utility_damage",
        2: "injured_or_dead_people",
        3: "missing_or_found_people",
        4: "not_humanitarian",
        5: "other_relevant_information",
        6: "rescue_volunteering_or_donation_effort",
        7: "vehicle_damage"
    }

    # Map predictions to labels using the label mapping
    label = [label_mapping[prediction] for prediction in y_pred]

    return {'img_class': label}


@app.get("/")
def root():
    return {'greeting': 'Hello'}

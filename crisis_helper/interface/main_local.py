import numpy as np
import pandas as pd
import random

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from crisis_helper.ml_logic.data import clean_data
from crisis_helper.ml_logic.preprocess import text_cleaning
from crisis_helper.ml_logic.model_binary import train_logistic_regression, predict_binary_logistic_regression
from crisis_helper.ml_logic.model_multiclass import train_logreg_multiclass, predict_logreg_multiclass
from crisis_helper.params import *
from sklearn.metrics import accuracy_score
from crisis_helper.ml_logic.registry import *


def preprocess_train_validate() -> None:
    """
    - Query the raw dataset from Kaggle API
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Save the model
    - Compute & save a validation performance metric
    """

    #print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    #min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    #max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    #query = f"""
     #   SELECT {",".join(COLUMN_NAMES_RAW)}
      #  FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
       # WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        #ORDER BY pickup_datetime
        #"""

    # Retrieve `query` data from Kaggle API or from `data_query_cache_path` if the file already exists!
    # Training dataset
    data_query_cache_train_path = Path(LOCAL_DATA_PATH).joinpath("raw", "df_train_binary.csv")
    data_query_cached_train_exists = data_query_cache_train_path.is_file()

    # Validation dataset
    data_query_cache_val_path = Path(LOCAL_DATA_PATH).joinpath("raw", "df_val_binary.csv")
    data_query_cached_val_exists = data_query_cache_val_path.is_file()

    if data_query_cached_train_exists and data_query_cached_val_exists:
        print("✅Loading data from local CSV...")

        df_train = pd.read_csv(data_query_cache_train_path)
        df_val = pd.read_csv(data_query_cache_val_path)

    else:
        print("✅Loading data from Kaggle API...")

        # $CODE_BEGIN
        #client = bigquery.Client(project=GCP_PROJECT)
        #query_job = client.query(query)
        #result = query_job.result()
        #data = result.to_dataframe()
        # $CODE_END

        # Save it locally to accelerate the next queries!
        #data.to_csv(data_query_cache_path, header=True, index=False)

    # Clean data using data.py
    df_train = clean_data(df_train)
    df_val = clean_data(df_val)

    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()

    # Fit the LabelEncoder to the 'label_text' column
    label_encoder = label_encoder.fit(df_train['label_text'])
    # Transform the train and test sets using the same label encoder
    df_train['encoded_label'] = label_encoder.transform(df_train['label_text'])
    df_val['encoded_label'] = label_encoder.transform(df_val['label_text'])

    # Preprocessing the column tweet_text using preprocess.py
    df_train['clean_texts'] = df_train.tweet_text.apply(text_cleaning)
    df_val['clean_texts'] = df_val.tweet_text.apply(text_cleaning)

    # Define X and y
    X_train_unvec = df_train['clean_texts']
    y_train = df_train['encoded_label']

    X_val_unvec = df_val['clean_texts']
    y_val = df_val['encoded_label']

    # Instantiate a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Train vectorizer on training data
    vectorizer.fit(X_train_unvec)

    # Transform the train and validation data
    X_train = vectorizer.transform(X_train_unvec)
    X_val = vectorizer.transform(X_val_unvec)

    # Train a model on the training set, using `model_binary.py` for binary classification
    model = train_logistic_regression(X_train, y_train)

    # Predict on the validation set
    y_pred_val_logreg = model.predict(X_val)

    # Evaluate the LogReg classifier on validation dataset
    accuracy_val_logreg = accuracy_score(y_val, y_pred_val_logreg)

    print(f"✅ Model trained on {X_train.shape[0]} rows with max val_accuracy: {round(accuracy_val_logreg, 2)}")

    # Save results, vectorizer and model
    save_results(metrics=dict(acc=accuracy_val_logreg))
    save_vectorizer(vectorizer)
    save_model(model=model)

    print("✅ Model,vectorizer and results saved")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    if X_pred is None:
        # Load test dataset from local path
        data_query_cache_test_path = Path(LOCAL_DATA_PATH).joinpath("raw", "df_test_binary.csv")
        data_query_cached_test_exists = data_query_cache_test_path.is_file()
        df_test = pd.read_csv(data_query_cache_test_path)
        X_pred = df_test["tweet_text"].iloc[random.randint(0, len(df_test) - 1)]
        X_pred = pd.DataFrame([X_pred], columns=["tweet_text"])

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

    print(f"Twit considered to be: {y_pred}")
    return y_pred


if __name__ == '__main__':
    try:
        preprocess_train_validate()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

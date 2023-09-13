import glob
import os
import time
import pickle
import sklearn

from colorama import Fore, Style
from sklearn.linear_model import LogisticRegression
#from google.cloud import storage
from crisis_helper.params import *
from tensorflow import keras


def save_results(params: dict=None, metrics: dict=None) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: sklearn.linear_model._logistic.LogisticRegression = None) -> None:

    """
    Saves model locally in a pickle file
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    #model.save(model_path)

    # Save model locally
    if model is not None:
        cwd = os.getcwd()
        models_path = os.path.join(cwd, LOCAL_REGISTRY_PATH, "text_model", timestamp + ".pickle")
        print(models_path)
        with open(models_path, 'wb') as file:
            pickle.dump(model, file)

        print("✅ Model saved locally")
        return None

def save_model_multiclass(model: sklearn.linear_model._logistic.LogisticRegression = None) -> None:

    """
    Saves model locally in a pickle file
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    #model.save(model_path)

    # Save model locally
    if model is not None:
        cwd = os.getcwd()
        models_path = os.path.join(cwd, LOCAL_REGISTRY_PATH, "text_model_multi", timestamp + ".pickle")
        print(models_path)
        with open(models_path, 'wb') as file:
            pickle.dump(model, file)

        print("✅ Model saved locally")
        return None


def save_vectorizer(vectorizer: sklearn.feature_extraction.text.TfidfVectorizer = None) -> None:

    """
    Saves vectorizer locally in a pickle file
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    #model.save(model_path)

    # Save vectorizer locally
    if vectorizer is not None:
        models_path = os.path.join(LOCAL_REGISTRY_PATH, "vectorizer", timestamp + ".pickle")
        with open(models_path, 'wb') as file:
            pickle.dump(vectorizer, file)

        print("✅ Vectorizer saved locally")
        return None


def load_model() -> sklearn.linear_model._logistic.LogisticRegression:
    #load_model(stage="Production") -> sklearn.linear_model._logistic.LogisticRegression:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    #if MODEL_TARGET == "local":
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "text_model")
    local_model_files = glob.glob(f"{local_model_directory}/*.pickle")

    if not local_model_files:
        return None

    most_recent_model_file_on_disk = sorted(local_model_files)[-1]

    # Open saved model
    with open(most_recent_model_file_on_disk ,'rb') as f:
        latest_model = pickle.load(f)

    print("✅ Model loaded from local disk")

    return latest_model


def load_model_multiclass() -> sklearn.linear_model._logistic.LogisticRegression:
    #load_model(stage="Production") -> sklearn.linear_model._logistic.LogisticRegression:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    #if MODEL_TARGET == "local":
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "text_model_multi")
    local_model_files = glob.glob(f"{local_model_directory}/*.pickle")

    if not local_model_files:
        return None

    most_recent_model_file_on_disk = sorted(local_model_files)[-1]

    # Open saved model
    with open(most_recent_model_file_on_disk ,'rb') as f:
        latest_model = pickle.load(f)

    print("✅ Model loaded from local disk")

    return latest_model


def load_vectorizer() -> sklearn.feature_extraction.text.TfidfVectorizer:
    #load_vectorizer(stage="Production") -> sklearn.feature_extraction.text.TfidfVectorizer:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    #if MODEL_TARGET == "local":
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_vectorizer_directory = os.path.join(LOCAL_REGISTRY_PATH, "vectorizer")
    local_vectorizer_files = glob.glob(f"{local_vectorizer_directory}/*.pickle")

    if not local_vectorizer_files:
        return None

    most_recent_vectorizer_file_on_disk = sorted(local_vectorizer_files)[-1]

   # Open saved vectorizer
    with open(most_recent_vectorizer_file_on_disk,'rb') as f:
        latest_vectorizer = pickle.load(f)

    print("✅ Vectorizer loaded from local disk")

    return latest_vectorizer


def load_img_model():

    #if MODEL_TARGET == "local":
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    # Get the latest model version name by the timestamp on disk
    local_img_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "img_model")
    local_img_model_files = glob.glob(f"{local_img_model_directory}/model_name.h5")

    if not local_img_model_files:
        return None

    most_recent_img_model_file_on_disk = sorted(local_img_model_files)[-1]
    latest_img_model = keras.models.load_model(most_recent_img_model_file_on_disk,
                                               compile=False)

    # Compile the model with the desired optimizer and settings
    latest_img_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])


   # Open saved img_model
    #with open(most_recent_img_model_file_on_disk,'rb') as f:
        #latest_img_model = pickle.load(f)


    print("✅ Image model loaded from local disk")

    return latest_img_model

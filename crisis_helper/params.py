import os
import numpy as np

##################  VARIABLES  ##################
'''DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "<your project id>" # TO COMPLETE
GCP_PROJECT_WAGON = "wagon-public-datasets"
BQ_DATASET = "taxifare"
BQ_REGION = "EU"
MODEL_TARGET = "local"'''
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".crisis_helper", "mlops", "data")
LOCAL_REGISTRY_PATH =  "pickle_files"

COLUMN_NAMES_RAW = ['event_name', 'tweet_id', 'image_id', 'tweet_text', 'image', 'label',
                    'label_text', 'label_image', 'label_text_image', 'new_image_path']

DTYPES_RAW = {
    'event_name': "object",
    'tweet_id': "int64",
    'image_id': "object",
    'tweet_text': "object",
    'image': "object",
    'label': "object",
    'label_text': "object",
    'label_image': "object",
    'label_text_image': "object",
    'new_image_path': "object"
}

#DTYPES_PROCESSED = np.float32

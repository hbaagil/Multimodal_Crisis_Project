FROM tensorflow/tensorflow:2.10.0

WORKDIR /prod

# First, pip install dependencies

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Then only, install crisis_helper!

COPY crisis_helper crisis_helper
COPY setup.py setup.py

RUN pip install .
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

COPY pickle_files pickle_files

CMD uvicorn crisis_helper.api.fast:app --host 0.0.0.0 --port $PORT

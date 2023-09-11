import string
import re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



def text_cleaning(X: pd.DataFrame):
    # Remove retweets like "RT @username:"
    sentence = re.sub(r'RT @\w+:', '', sentence)

    # Removing whitespaces
    sentence = sentence.strip()

    # Lowercasing
    sentence = sentence.lower()

    # Removing numbers and unwanted characters
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)

    # Replace URLs with <URL> placeholder
    sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence)

    # Tokenizing
    tokenized = word_tokenize(sentence)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokenized if word not in stop_words]

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
    cleaned_sentence = " ".join(lemmatized)

    print("✅ feature text preprocessed")

    return cleaned_sentence


"""def tfidf_vectorizer():

    Instantiate TF-IDF vectorizer.


    # Instantiate a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    print("✅ feature text with TF-IDF vectorized")
    return vectorizer

def tfidf_vectorizer_fit(df, vectorizer):
    fit_feature = vectorizer.fit(df)

    return fit_feature

def tfidf_vectorizer_transform(df, vectorizer):
    fit_feature = vectorizer.fit(df)

    return fit_feature"""

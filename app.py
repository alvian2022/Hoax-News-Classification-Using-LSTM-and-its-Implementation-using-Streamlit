import time
import streamlit as st
import numpy as np
import pandas as pd
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
import requests
import os
import string
import tensorflow
from tensorflow import keras

import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

from PIL import Image


st.set_page_config(page_title="Indonesian News Title Category Classifier",
                   page_icon="🗞️", layout="centered")


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=3600, max_entries=10)
def build_model():
    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        model = keras.models.load_model('LSTM.h5')
        with open('tokenizer.pickle', 'rb') as f:
            Tokenizer = pickle.load(f)

        inference = model, Tokenizer
    return inference


inference, Tokenizer = build_model()
model_baseline = keras.models.load_model('LSTM_baseline.h5')

st.title('🗞️ Indonesian News Title Category Classifier')

with st.expander('📋 About this app', expanded=True):
    st.markdown("""
    * Indonesian News Title Category Classifier app is an easy-to-use tool that allows you to predict the category of a given news title.
    * You can predict one title at a time or upload .csv file to bulk predict.
    * Made by [Alpian Khairi](https://www.linkedin.com/in/alpiankhairi/), [Sheva Satria](), [Fernandico](), [Bagus]().
    """)
    st.markdown(' ')

with st.expander('🧠 About prediction model', expanded=False):
    st.markdown("""
    ### Indonesian News Title Category Classifier
    * Model are trained using [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) based on [Indonesian News Title Dataset]() from  on .
    * Model test accuracy is **~93%**.
    * **[Source Code]()**
    """)
    st.markdown(' ')

st.markdown('### LSTM Algorithm Modified')
image_lstm = Image.open('image/output.png')
st.image(image_lstm, caption='Model Accuracy')

st.markdown('### LSTM Algorithm Baseline')
image_lstm = Image.open('image/output_baseline.png')
st.image(image_lstm, caption='Model Accuracy')


st.markdown(' ')
st.markdown(' ')

st.header('🔍 News Title Category Prediction')

title = st.text_input(
    'News Title', placeholder='Enter your shocking news title + its narrative')

if title:
    with st.spinner('Loading prediction...'):
        tokenized_test = Tokenizer.texts_to_sequences(title)
        X_test = pad_sequences(tokenized_test, maxlen=250)
        result = inference.predict(X_test)
        result_baseline = model_baseline.predict(X_test)
        pred_labels = []
        pred_labels_baseline = []
        for i in result:
            if i > 0.7:
                pred_labels.append(1)
                pred_labels_baseline.append(1)
            else:
                pred_labels.append(0)
                pred_labels_baseline.append(0)

        for i in range(len(title)):
            if pred_labels[i] == 1:
                s = 'Fact'
            else:
                s = 'Hoax'

        for i in range(len(title)):
            if pred_labels_baseline[i] == 1:
                s = 'Fact'
            else:
                s = 'Hoax'
    st.markdown(
        f'Category for this news is **[{s}]** based on model modification')
    st.markdown(f'Category for this news is **[{s}]** based on baseline model')


st.markdown(' ')
st.markdown(' ')

st.header('🗃️ Bulk News Title Category Prediction')
st.markdown(
    'Only upload .csv file that contains list of news titles separated by comma.')

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.head(100)
    results = []
    results_baseline = []
    # df = df.tail(20)

    with st.spinner('Loading prediction...'):
        for title in df['narasi']+df['judul']:
            tokenized_test = Tokenizer.texts_to_sequences(title)
            X_test = pad_sequences(tokenized_test, maxlen=250)
            result = inference.predict(X_test)
            result_baseline = model_baseline.predict(X_test)

            pred_labels = []
            pred_labels_baseline = []
            for i in result:
                if i > 0.7:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            for i in result_baseline:
                if i > 0.7:
                    pred_labels_baseline.append(1)
                else:
                    pred_labels_baseline.append(0)

            for i in range(len(title)):
                if pred_labels[i] == 1:
                    s = 'Hoax'
                else:
                    s = 'Fact'

            for i in range(len(title)):
                if pred_labels_baseline[i] == 1:
                    s_baseline = 'Hoax'
                else:
                    s_baseline = 'Fact'

            results.append({'Title': title, 'Category': s})
            results_baseline.append({'Title': title, 'Category': s_baseline})

        df_results = pd.DataFrame(results)
        df_results_baseline = pd.DataFrame(results_baseline)

    st.markdown('#### Prediction Result by Modified Model')
    st.download_button(
        "Download Result",
        df_results.to_csv(index=False).encode('utf-8'),
        "News Title Category Prediction Result.csv",
        "text/csv",
        key='download-csv'
    )
    st.dataframe(df_results, 100)

    st.markdown('#### Prediction Result by Baseline Model')
    st.download_button(
        "Download Result",
        df_results_baseline.to_csv(index=False).encode('utf-8'),
        "News Title Category Prediction Result Baseline.csv",
        "text/csv",
        key='download-csv'
    )
    st.dataframe(df_results_baseline, 100)

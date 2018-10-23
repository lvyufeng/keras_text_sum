import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_INPUT_SEQ_LENGTH = 500
MAX_OUTPUT_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_OUTPUT_VOCAB_SIZE = 2000

def load_data(path = "./../../datasets/fake_or_real_news.csv"):
    df = pd.read_csv(path)
    X = df['text']
    Y = df['title']

    # X data
    X_tokenizer = Tokenizer(num_words=MAX_INPUT_VOCAB_SIZE)
    X_tokenizer.fit_on_texts(X)
    X_sequences = X_tokenizer.texts_to_sequences(X)

    X_word_index = X_tokenizer.word_index

    X_data = pad_sequences(X_sequences, MAX_INPUT_SEQ_LENGTH,padding='pre')

    # Y_data
    Y_tokenizer = Tokenizer(num_words=MAX_OUTPUT_VOCAB_SIZE)
    Y_tokenizer.fit_on_texts(Y)
    Y_sequences = Y_tokenizer.texts_to_sequences(Y)
    #
    # Y_word_index = Y_tokenizer.word_index
    # print(X_word_index['you'],Y_word_index['you'])
    # Y_data = pad_sequences(Y_sequences, MAX_OUTPUT_SEQ_LENGTH,padding='post')
    # Y_data_final = np.zeros((len(Y), MAX_OUTPUT_SEQ_LENGTH, MAX_OUTPUT_VOCAB_SIZE))
    #
    # for line_index, word_indexes in enumerate(Y_data):
    #     for list_index, word_index in enumerate(word_indexes):
    #         Y_data_final[line_index, list_index, word_index] = 1
    #         pass


    return X_data,Y

def load_np_data(path = "./../../datasets/fake_or_real_news.csv"):

    df = pd.read_csv(path)
    X = df['text']
    Y = df['title']

    # X data
    X_tokenizer = Tokenizer(num_words=MAX_INPUT_VOCAB_SIZE)
    X_tokenizer.fit_on_texts(X)
    X_sequences = X_tokenizer.texts_to_sequences(X)

    X_word_index = X_tokenizer.word_index

    X_data = pad_sequences(X_sequences, MAX_INPUT_SEQ_LENGTH,padding='pre')

    # Y_data
    Y_tokenizer = Tokenizer(num_words=MAX_OUTPUT_VOCAB_SIZE)
    Y_tokenizer.fit_on_texts(Y)
    Y_sequences = Y_tokenizer.texts_to_sequences(Y)

    Y_word_index = Y_tokenizer.word_index

    Y_data = pad_sequences(Y_sequences, MAX_OUTPUT_SEQ_LENGTH,padding='pre')
    Y_data_final = np.zeros((len(Y),MAX_OUTPUT_SEQ_LENGTH,MAX_OUTPUT_VOCAB_SIZE))

    for line_index, word_indexes in enumerate(Y_data):
        for list_index, word_index in enumerate(word_indexes):
            Y_data_final[line_index,list_index,word_index] = 1


    X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data_final,test_size=0.2,random_state=42)

    # param = get_param(X, Y)

    return X_train, X_test, Y_train, Y_test

# load_np_data()
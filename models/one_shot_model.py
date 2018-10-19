# import keras
from keras.layers import Input,Embedding,LSTM,RepeatVector,TimeDistributed,Dense
from keras.models import Model
import os
import numpy as np

class OneShotModel():

    def __init__(self, src_vocab_size, sum_vocab_size, src_txt_length, sum_txt_length):
        # pass
        self.src_vocab_size = src_vocab_size
        self.sum_vocab_size = sum_vocab_size
        self.src_txt_length = src_txt_length
        self.sum_txt_length = sum_txt_length


        # encoder input model
        inputs = Input(shape=(src_txt_length,))
        encoder1 = Embedding(src_vocab_size,128)(inputs)
        encoder2 = LSTM(128)(encoder1)
        encoder3 = RepeatVector(sum_txt_length)(encoder2)
        # decoder output model
        decoder1 = LSTM(128,return_sequences=True)(encoder3)
        outputs = TimeDistributed(Dense(sum_vocab_size,activation='softmax'))(decoder1)

        # tie it together

        model = Model(inputs=inputs,outputs=outputs)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        self.model = model

    def fit(self, X_train, Y_train, X_test, Y_test, epochs = 1, batch_size = 32, model_save_path = None):
        history = self.model.fit(X_train,Y_train,batch_size,epochs,
                       validation_data=(X_test, Y_test))
        self.model.save_weights(model_save_path)
        return history


    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_weights(path)
        else:
            print('There is no weights files')

    def summarize(self, src, word_index = None):
        predicted = self.model.predict(src)
        predicted_word_index = np.argmax(predicted,axis=2)

        return predicted_word_index
        # pass
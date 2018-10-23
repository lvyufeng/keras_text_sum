from keras.layers import Input,Embedding,LSTM,RepeatVector,TimeDistributed,Dense,Bidirectional,Flatten,Multiply,Activation,Permute,Add
from keras.models import Model
import os
import numpy as np

class LSTM_ATT_Model():
    def __init__(self, src_vocab_size, sum_vocab_size, src_txt_length, sum_txt_length, embedding_dim = 128, embedding_matrix = None):

        # encoder
        encoder_inputs = Input(shape=(src_txt_length,))
        # embedding_sequenses = Embedding(sum_vocab_size, embedding_dim, weights= [embedding_matrix],input_length=src_txt_length,trainable= False)(encoder_inputs)
        encoder_embedding = Embedding(src_txt_length, embedding_dim,
                                        input_length=src_txt_length, trainable=True)(encoder_inputs)
        encoder_LSTM = LSTM(128,dropout=0.2,return_sequences=True,return_state=True)
        encoder_LSTM_rev = LSTM(128,dropout=0.2,return_sequences=True,return_state=True,go_backwards=True)

        # encoder_LSTM = Bidirectional(LSTM(128, dropout_W=0.1, dropout_U=0.1, return_sequences=True,return_state=True))
        # bilstm_d = Dropout(0.1)(bilstm)
        encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
        encoder_outputs_rev, state_h_rev, state_c_rev = encoder_LSTM_rev(encoder_embedding)

        encoder_outputs = Add()([encoder_outputs,encoder_outputs_rev])

        state_h = Add()([state_h,state_h_rev])
        state_c = Add()([state_c,state_c_rev])
        encoder_states = [state_h,state_c]

        # decoder

        decoder_inputs = Input(shape=(sum_txt_length,))
        decoder_embedding = Embedding(sum_vocab_size, embedding_dim,
                                        input_length=sum_txt_length, trainable=True)(decoder_inputs)
        decoder_LSTM = LSTM(128,return_sequences=True,dropout_U = 0.2, dropout_W = 0.2, return_state=True)

        decoder_outputs, _, _ = decoder_LSTM(decoder_embedding,initial_state=encoder_states)

        # attention

        attention = TimeDistributed(Dense(1,activation='tanh'))(encoder_outputs)
        attention = Flatten()(attention)
        attention = Dense(128,activation='tanh')(attention)
        attention = Multiply()([decoder_outputs,attention])
        attention = Activation('softmax')(attention)
        attention = Permute([2,1])(attention)

        decoder_dense = Dense(sum_txt_length,activation='softmax')
        decoder_outputs = decoder_dense(attention)


        model = Model(inputs = [encoder_inputs,decoder_inputs],outputs = decoder_outputs)

        print(model.summary())


        pass


MAX_INPUT_SEQ_LENGTH = 500
MAX_OUTPUT_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_OUTPUT_VOCAB_SIZE = 2000



model = LSTM_ATT_Model(MAX_INPUT_VOCAB_SIZE,MAX_OUTPUT_VOCAB_SIZE,MAX_INPUT_SEQ_LENGTH,MAX_OUTPUT_SEQ_LENGTH)



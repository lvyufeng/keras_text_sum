import keras
from keras.layers import Input,Embedding,LSTM,RepeatVector,TimeDistributed,Dense
from keras.models import Model


vocab_size = ''
src_txt_length = ''
sum_txt_length = ''


# encoder input model
inputs = Input(shape=(src_txt_length,))
encoder1 = Embedding(vocab_size,128)(inputs)
encoder2 = LSTM(128)(encoder1)
encoder3 = RepeatVector(sum_txt_length)(encoder2)
# decoder output model
decoder1 = LSTM(128,return_sequences=True)(encoder3)
outputs = TimeDistributed(Dense(vocab_size,activation='softmax'))(decoder1)

# tie it together

model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='categorical_crossentropy',optimizer='adam')


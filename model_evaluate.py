from utils.data_loaders.fake_or_real_news_loader import load_np_data
from models import OneShotModel


MAX_INPUT_SEQ_LENGTH = 500
MAX_OUTPUT_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_OUTPUT_VOCAB_SIZE = 2000


X_train, X_test, Y_train, Y_test = load_np_data(path='./datasets/fake_or_real_news.csv')
print(X_train.shape)
print(Y_train.shape)

model = OneShotModel(MAX_INPUT_VOCAB_SIZE,MAX_OUTPUT_VOCAB_SIZE,MAX_INPUT_SEQ_LENGTH,MAX_OUTPUT_SEQ_LENGTH)

model.load_weights('./models_save/one_shot_model-weights.h5')


result = model.evaluate(X_train,Y_train)
print(result)
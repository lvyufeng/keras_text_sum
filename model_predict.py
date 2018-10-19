from utils.data_loaders.fake_or_real_news_loader import load_data
from models import OneShotModel


MAX_INPUT_SEQ_LENGTH = 500
MAX_OUTPUT_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_OUTPUT_VOCAB_SIZE = 2000


X,Y = load_data(path='./datasets/fake_or_real_news.csv')

model = OneShotModel(MAX_INPUT_VOCAB_SIZE,MAX_OUTPUT_VOCAB_SIZE,MAX_INPUT_SEQ_LENGTH,MAX_OUTPUT_SEQ_LENGTH)

model.load_weights('./models_save/one_shot_model-weights.h5')

for i in range(len(Y)):
    x = X[i]
    y = Y[i]
    y_predict = model.summarize(x)
    pass



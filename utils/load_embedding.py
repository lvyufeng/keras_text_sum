import numpy as np


def load_embedding(path):
    embedding_index = {}
    f = open(path,encoding='utf8')
    for index,line in enumerate(f):
        if index == 0:
            continue
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    return embedding_index


load_embedding('/home/lv/data_set/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt')
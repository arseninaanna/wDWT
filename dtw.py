import numpy as np
import nltk
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from prepare_data import tokenize

nltk.download('punkt')

# please, ensure that you have downloaded google word2vec from
# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
# and elmo embedding from https://tfhub.dev/google/elmo/2?tf-hub-format=compressed
elmo = hub.Module("elmo_dir/", trainable=False)
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


def split_into_pars(text):
    return text.split('\n')


def embedding_random(words, size=150):
    # random embedding for testing
    answer = {word: np.random.rand(size) for word in words}
    return list(answer[word] for word in words)


def embedding_elmo(data):
    # pretrained elmo embedding (context meaning of each word in a sentence)
    # embedding size is 1024
    sentences = [nltk.word_tokenize(sentence) for sentence in data]
    lens = [len(sentence) for sentence in sentences]
    embeddings = elmo(data, signature="default", as_dict=True)["elmo"]
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        elmo_embeddings = session.run(embeddings)
    embs = list()
    for index, emb in enumerate(elmo_embeddings):
        i = 0
        while i < lens[index]:
            embs.append(emb[i])
            i += 1
            if i >= len(emb):
                break
    return np.array(embs)


def embedding_word2vec(words):
    # word2vec embedding from google (vector size = 300)
    valid_words = [word for word in words if word in word2vec.vocab.keys()]
    # print(f'{len(words)} -> {len(valid_words)}')
    return [word2vec[word] for word in valid_words]


def DistPara(p, q):
    # calculates DTW_para
    e = len(p)
    f = len(q)
    DTW = np.empty(shape=(e + 1, f + 1))
    DTW[0] = np.array([float('inf') for i in range(DTW.shape[1])])
    DTW[:, 0] = np.array([float('inf') for i in range(DTW.shape[0])])
    DTW[0][0] = 0
    for i in range(1, e + 1):
        for j in range(1, f + 1):
            distance = np.linalg.norm(p[i - 1] - q[j - 1])
            DTW[i][j] = distance + min(DTW[i - 1][j], DTW[i][j - 1], DTW[i - 1][j - 1])
    return DTW[e][f]


def wDTW_docs(d1, d2):
    # calculates DTW_doc
    m = len(d1)
    n = len(d2)
    wDTW = np.empty(shape=(m + 1, n + 1))
    wDTW[0] = np.array([float('inf') for i in range(wDTW.shape[1])])
    wDTW[:, 0] = np.array([float('inf') for i in range(wDTW.shape[0])])
    wDTW[0][0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            distance = DistPara(d1[i - 1], d2[j - 1])
            wDTW[i][j] = distance + min(wDTW[i - 1][j], wDTW[i][j - 1], wDTW[i - 1][j - 1])
    return wDTW[m][n]


def final_distance(text1, text2, embedding=embedding_word2vec):
    pars1 = split_into_pars(text1)
    pars2 = split_into_pars(text2)
    pars1 = [embedding(tokenize(par)) for par in pars1]
    pars2 = [embedding(tokenize(par)) for par in pars2]
    return wDTW_docs(pars1, pars2)

import numpy as np
import nltk
import tensorflow as tf 
# use tensorflow v. 1.x
import tensorflow_hub as hub
import re
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import prepare_data
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import os
nltk.download('punkt')

# please, ensure that you have downloaded google word2vec from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
# and elmo embedding from https://tfhub.dev/google/elmo/2?tf-hub-format=compressed
elmo = hub.Module("elmo_dir/", trainable=False)
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

def get_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory + "/" + filename, encoding="utf-8-sig")
            lines = f.read()
            documents.append(lines)
    return documents

def tokenize(text):
  tmp = text.translate(str.maketrans('', '', string.punctuation))
  return word_tokenize(tmp.lower())

def split_into_pars(text):
  return text.split('\n')

def embedding_random(words, size=15):
  # random embedding for testing
  answer = {word: np.random.rand(size) for word in words}
  return list(random_embedding[word] for word in words)


def embedding_word2vec(words):
  # word2vec embedding from google (vector size = 300)
  valid_words = [word for word in words if word in word2vec.vocab.keys()]
  # print(f'{len(words)} -> {len(valid_words)}')
  return [word2vec[word] for word in valid_words]


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


def DistPara(p, q):
  # calculates DTW_para
  e = len(p)
  f = len(q)
  DTW = np.empty(shape=(e + 1, f + 1))
  DTW[0] = np.array([float('inf') for i in range(DTW.shape[1])])
  DTW[:, 0] = np.array([float('inf') for i in range(DTW.shape[0])])
  DTW[0][0] = 0
  Dist = np.zeros(shape=())
  for i in range(1, e + 1):
    for j in range(1, f + 1):
      distance = np.linalg.norm(p[i-1] - q[j-1])
      DTW[i][j] = distance + min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
  return DTW[e][f]


def wDTW_docs(d1, d2):
  # calculates DTW_doc
  m = len(d1)
  n = len(d2)
  wDTW = np.empty(shape=(m + 1, n + 1))
  wDTW[0] = np.array([float('inf') for i in range(wDTW.shape[1])])
  wDTW[:, 0] = np.array([float('inf') for i in range(wDTW.shape[0])])
  wDTW[0][0] = 0
  Dist = np.zeros(shape=())
  for i in range(1, m + 1):
    for j in range(1, n + 1):
      distance = DistPara(d1[i-1], d2[j-1])
      wDTW[i][j] = distance + min(wDTW[i-1][j], wDTW[i][j-1], wDTW[i-1][j-1])
  return DTW[m][n]

def final_distance(text1, text2, embedding=embedding_word2vec):
  pars1 = split_into_pars(text1)
  pars2 = split_into_pars(text2)
  pars1 = [embedding(tokenize(par)) for par in pars1]
  pars2 = [embedding(tokenize(par)) for par in pars2]
  return wDTW_docs(pars1, pars2)


corpus = get_documents("/content/pan-plagiarism-corpus-2011/external-detection-corpus/source-document/part1")
# preprocessing and doc2vec model training
prepared_docs = []
for document in corpus:
        tmp = document.translate(str.maketrans('', '', string.punctuation))
        prepared_docs.append(word_tokenize(tmp.lower()))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(prepared_docs)]

max_epochs = 10
vec_size = 1000
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=alpha/100,
                min_count=1,
                dm = 1)
print("Model created")
  
model.build_vocab(documents)
print("Vocabulary builded")

for epoch in range(max_epochs):
    print(f'iteration {epoch}')
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("d2v-1000.model")
print("Model Saved")

# test the model on dataset

f = open("/content/pan-plagiarism-corpus-2011/external-detection-corpus/source-document/part1/source-document00001.txt", encoding="utf-8-sig")
corpus_source = f.read()
max_sim = -10000
max_vv = ""
for i in range(1, 501):
  ind = "00000"
  vv = str(i)
  if i%10 == 1:
    print(f"Document in processing: {i}")
  vv = (len(ind) - len(vv))*'0' + vv
  f = open(f"/content/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document/part1/suspicious-document{vv}.txt", encoding="utf-8-sig")
  corpus_susp = f.read()
  curr = model.docvecs.similarity_unseen_docs(model, tokenize(corpus_source), tokenize(corpus_susp))
  if curr > max_sim:
    max_sim = curr
    max_vv = vv
print(f"{max_vv} : {max_sim}")

# test the model on a custom texts
corp1 = 'На международных конференциях в Циммервальде (1915) и Кинтале (1916) Ленин, в соответствии с резолюцией Штутгартского конгресса и Базельским манифестом II Интернационала, отстаивал свой тезис о необходимости превращения империалистической войны в войну гражданскую и выступал с лозунгом «революционного пораженчества»: одинакового желания поражения в бессмысленной для народа, который в случае победы останется в таком же угнетённом положении, братоубийственной ради прибыли монополий и рынков сбыта империалистической войне — как собственной стране, так и её противнику, так как крах буржуазной власти создаёт революционную ситуацию и открывает возможности трудящимся защищать свои интересы, а не интересы своих угнетателей и создать более справедливый общественный строй как в своей стране, так и в стране-противнике'
сorp2 = 'На международных конференциях в Циммервальде (1915 г.) и Кинтале (1916 г.) Ленин в соответствии с резолюцией Штутгартского конгресса и Базельским манифестом II Интернационала защищал свой тезис о необходимости превращения империалистической войны в гражданскую и выступал за лозунг «революционного пораженчества»: но стремление к победе не имеет смысла для народа, который в случае победы останется на том же угнетенном положении, братоубийственно на благо монополий и рынков империалистической войны, подобных их собственная страна и ее враг, потому что крах буржуазной власти создает революционную ситуацию и открывает возможности для рабочих защищать свои интересы, а не интересы своих угнетателей, и создавать более справедливую социальную систему как в своей стране, так и в стране. вражеская страна'
corp3 = 'В 2017 году Комиссия по борьбе с лженаукой и фальсификацией научных исследований при Президиуме РАН выпустила меморандум, который признаёт гомеопатию лженаукой. В меморандуме изложены рекомендации, направленные на исключение гомеопатии из системы российского здравоохранения. В рамках меморандума комиссия предложила Министерству здравоохранения РФ исключить медицинское употребление гомеопатии в муниципальных и государственных лечебных учреждениях, а также рекомендовала аптекам не продавать гомеопатические и лекарственные препараты совместно'
print(f"{model.docvecs.similarity_unseen_docs(model, tokenize(corp1), tokenize(corp2))} for close docs and {model.docvecs.similarity_unseen_docs(model, tokenize(corp1), tokenize(corp3))} for different")




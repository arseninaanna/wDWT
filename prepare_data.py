import numpy as np
import os
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize(text):
    tmp = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(tmp.lower())


def get_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory + "/" + filename, encoding="utf-8-sig")
            lines = f.read()
            documents.append(lines)
    return documents


def doc2vec(documents, vector_size=15):
    vectors = []
    all_pargraphs = []
    doc_to_paragraphs = {}
    counter = 0

    for document in documents:
        paragraphs = document.split('\n\n')

        doc_to_paragraphs[counter] = len(paragraphs)
        counter += 1
        for par in paragraphs:
            all_pargraphs.append(par.lower())

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_pargraphs)]
    model = Doc2Vec(documents, vector_size=vector_size, window=2, min_count=1, workers=4)

    start = 0
    for index in doc_to_paragraphs:
        length = doc_to_paragraphs[index]
        doc_vectors = []
        for i in range(start, length + start):
            doc_vectors.append(model.docvecs[i])
        vectors.append(doc_vectors)
        start = start + length

    return np.array(vectors)

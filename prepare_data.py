import numpy as np
import os
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def preprocess(directory, vecor_size = 15):
    vectors = []
    all_pargraphs = []
    doc_to_paragraphs = {}
    counter = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            f = open(directory + "/" + filename, encoding="utf-8-sig")
            lines = f.read()
            paragraphs = lines.split('\n\n')

            doc_to_paragraphs[counter] = len(paragraphs)
            counter += 1
            for par in paragraphs:
                all_pargraphs.append(par.lower())

            continue
        else:
            continue

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_pargraphs)]
    model = Doc2Vec(documents, vector_size=vecor_size, window=2, min_count=1, workers=4)
    print(model.docvecs[0])

    start = 0
    for index in doc_to_paragraphs:
        length = doc_to_paragraphs[index]
        doc_vectors = []
        for i in range(start, length+start):
            doc_vectors.append(model.docvecs[i])
        vectors.append(doc_vectors)
        start = start+length

    return np.array(vectors)


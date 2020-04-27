import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize(text):
    tmp = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(tmp.lower())


def read_pair(susp, src):
    src_file = open("./data/pan12-text-alignment/src/" + src, "r")
    src_data = src_file.readlines()
    src_file.close()

    susp_file = open("./data/pan12-text-alignment/susp/" + susp, "r")
    susp_data = susp_file.readlines()
    susp_file.close()

    return tuple([src_data, susp_data])


def get_documents(N=50):
    no_plagiarism = []
    plagiarism = []

    with open("./data/pan12-text-alignment/01_no_plagiarism/pairs", "r") as file:
        for i in range(N):
            line = next(file).strip()
            susp, source = line.split()
            no_plagiarism.append(read_pair(susp, source))

    with open("./data/pan12-text-alignment/02_no_obfuscation/pairs", "r") as file:
        for i in range(N):
            line = next(file).strip()
            susp, source = line.split()
            plagiarism.append(read_pair(susp, source))

    return no_plagiarism, plagiarism


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

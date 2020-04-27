from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from prepare_data import get_documents, tokenize
import string
from nltk.tokenize import word_tokenize

max_epochs = 20
vec_size = 1000
alpha = 0.025

def train_doc2vec(pairs, max_epochs=20, vec_size=1000, alpha=0.025):
    data = [x[0] for x in pairs]
    # doc2vec model training
    prepared_docs = []
    for document in data:
        tmp = document.translate(str.maketrans('', '', string.punctuation))
        prepared_docs.append(word_tokenize(tmp.lower()))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(prepared_docs)]



    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=alpha / 100,
                    min_count=1,
                    dm=1)
    print("Model is created")

    model.build_vocab(documents)
    print("Vocabulary is built")

    for epoch in range(max_epochs):
        print(f'iteration {epoch}')
        model.train(documents,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save(f"d2v-{vec_size}.model")
    print("Model Saved")
    return model


def test_doc2vec(corpus):
    corpus_source = [x[0] for x in corpus[0]]
    max_sim = -10000
    max_vv = ""
    for i in range(len(corpus[0])):
        ind = "00000"
        vv = str(i)
        if i % 10 == 1:
            print(f"Document in processing: {i}")
        vv = (len(ind) - len(vv)) * '0' + vv
        f = open(
            f"./data/pan-plagiarism-corpus-2011/external-detection-corpus/suspicious-document/part1/suspicious-document{vv}.txt",
            encoding="utf-8-sig")
        corpus_susp = f.read()
        curr = model.docvecs.similarity_unseen_docs(model, tokenize(corpus_source), tokenize(corpus_susp))
        if curr > max_sim:
            max_sim = curr
            max_vv = vv
    print(f"{max_vv} : {max_sim}")


def apply_doc2vec(text1, text2):
    model = Doc2Vec.load(f"d2v-{vec_size}.model")
    return model.docvecs.similarity_unseen_docs(model, tokenize(text1), tokenize(text2))


if __name__ == "__main__":
    model = None
    if open(f"d2v-{vec_size}.model"):
        model = Doc2Vec.load(f"d2v-{vec_size}.model")
    else:
        corpus = get_documents(500)
        model = train_doc2vec(corpus[0])
    assert model != None, 'model is broken'
    # testing code:
    # test_doc2vec(corpus)
    '''
    corp1 = "During the first conference, Lenin spoke to the delegates, lambasting the parliamentary path to socialism espoused by revisionist Marxists like Kautsky and repeating his calls for a violent overthrow of Europe's bourgeoisie governments."
    corp2 = "Lenin's predicted world revolution did not materialise, as the Hungarian communist government was overthrown and the German Marxist uprisings suppressed."
    corp3 = "Founded by Ken Morrison and Dave Thomas in the early 1970s, the company, then known as Ocean Sound, originally started as a North Vancouver-based music recording studio. Musicians such as Art Bergmann, Chilliwack, D.O.A and k.d. lang recorded albums and songs there."
    print(
    f"{model.docvecs.similarity_unseen_docs(model, tokenize(corp1), tokenize(corp2))} for close docs and {model.docvecs.similarity_unseen_docs(model, tokenize(corp1), tokenize(corp3))} for different")
    '''

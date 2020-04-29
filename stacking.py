from dtw import *
from prepare_data import get_documents
from doc2vec_model import apply_doc2vec
model_thresholds = list()


def calculate_accuracy(model, threshold=1.5, N_for_estimation=50):
    dataset = get_documents(N_for_estimation)
    answers = list()
    for not_copied_pair in dataset[0]:
        answers.append(final_distance(not_copied_pair[0], not_copied_pair[1], embedding=model) > threshold)
    for copied_pair in dataset[1]:
        answers.append(final_distance(model, *copied_pair) <= threshold)
    return [answers[i] != i > N_for_estimation for i in range(len(answers))].count(True)/(2*N_for_estimation)


def find_optimal_threshold(model, iter_size=300, iters=25):
    search_space = np.linspace(0, 1000, iter_size)
    t_n = 0
    for i in range(iters):
        maxx = -float('inf')
        ans = -1
        for j in search_space:
            t_n += 1
            print(t_n)
            accuracy = calculate_accuracy(model, threshold=j)
            if accuracy > maxx:
                maxx = accuracy
                ans = j
        if ans == -1 or iter_size:
            return search_space[-1]
        else:
            search_space = np.linspace(search_space[ans], search_space[ans+1], iter_size)
    return search_space[-1]


def vote_model(text1, text2, thresholds=None):
    if thresholds is None:
        thresholds = model_thresholds
    answers = list()
    models = [embedding_word2vec, embedding_elmo]
    for i in range(len(models)):
        answers.append(final_distance(text1, text2, models[i]) > thresholds[i])
    answers.append(apply_doc2vec(text1, text2) > thresholds[-1])
    if answers.count(True) > answers.count(False):
        answer = True
    else:
        answer = False
    return answer


if __name__ == "__main__":
    for model in [embedding_word2vec, embedding_elmo]:
        model_thresholds.append(find_optimal_threshold(model))
        print(model_thresholds)



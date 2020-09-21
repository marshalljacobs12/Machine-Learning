import numpy as np

from util import accuracy
from hmm import HMM
from data_process import Dataset


def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    ###################################################
    obs_dict = {}
    state_dict = {}
    curr_index = 0
    for tag in tags:
        state_dict[tag] = curr_index
        curr_index += 1

    curr_index = 0
    for line in train_data:
        for word in line.words:
            if word not in obs_dict:
                obs_dict[word] = curr_index
                curr_index += 1

    S = len(state_dict.keys())
    L = len(obs_dict.keys())
    pi = np.zeros([S])
    A = np.zeros([S, S])
    B = np.zeros([S, L])

    for line in train_data:
        pi[state_dict[line.tags[0]]] += 1
    pi /= np.sum(pi)

    for line in train_data:
        for i in range(len(line.tags)-1):
            A[state_dict[line.tags[i]], state_dict[line.tags[i+1]]] += 1

    for i in range(S):
        A[i, :] /= np.sum(A[i, :])

    for line in train_data:
        for i in range(len(line.words)):
            B[state_dict[line.tags[i]], obs_dict[line.words[i]]] += 1

    for i in range(S):
        B[i, :] /= np.sum(B[i, :])

    model = HMM(pi, A, B, obs_dict, state_dict)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    ###################################################
    S = len(model.state_dict.keys())
    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict:
                b = np.ones([S, 1]) * 1e-6
                model.B = np.append(model.B, b, axis=1)
                model.obs_dict[word] = len(model.obs_dict.keys())

    for line in test_data:
        tagged_sentence = model.viterbi(line.words)
        tagging.append(tagged_sentence)

    return tagging

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from collections import OrderedDict
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import scipy.stats
import json

POS_LIST = ['"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
NER_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


def Rob(model, advx, advy):
    """ Robustness (empirical) 
    args:
        model: suspect model
        advx: black-box test cases (adversarial examples) 
        advy: ground-truth labels
    
    return:
        Rob value
    """
    return round(np.sum(np.argmax(model.predict(advx), axis=1) == np.argmax(advy, axis=1)) / advy.shape[0], DIGISTS)


def JSD(model1, model2, advx):
    """ Jensen-Shanon Distance
    args:
        model1 & model2: victim model and suspect model
        advx: black-box test cases 
    return:
        JSD value
    """
    DIGISTS = 4
    vectors1 = model1.predict(advx)
    vectors2 = model2.predict(advx)
    mid = (vectors1 + vectors2) / 2
    distances = (scipy.stats.entropy(vectors1, mid, axis=1) + scipy.stats.entropy(vectors2, mid, axis=1)) / 2
    return round(np.average(distances), DIGISTS)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def transformations(dataframe):
    # upper to lower character
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # #punctuations
    # dataframe['text'] = dataframe['text'].str.replace('[^\w\s]','')
    # #numbers
    # dataframe['text'] = dataframe['text'].str.replace('\d','')
    # #remove stop words
    # sw = stopwords.words('english')
    # dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    # #rare characters deleting
    # sil = pd.Series(' '.join(dataframe['text']).split()).value_counts()[-1000:]
    # dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
    # #lemmi
    # from textblob import Word
    # #nltk.download('wordnet')
    # dataframe['text'] = dataframe['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return dataframe


def build_periodogram(xy_array, n_freqs=200000, k=0.5):
    '''
    :param xy_array:
    :param labels:
    :param N:
    :param n_freqs:
    :return:

    The returned array has the frequencies as its first column, and the remaining blocks
    of N columns represent the softmax scores of the inputs of a given label i, for the logit
    corresponding to index j
    '''
    freqs_array = np.zeros((n_freqs, 2))
    freqs = np.linspace(0.002, 40, n_freqs)
    freqs_array[:, 0] = freqs

    x = xy_array[:, 0]
    if x.shape[0] == 0:
        freqs_array[:, 1] = np.zeros(n_freqs)
        thetas = [0.0, 0.0, 0.0]
    else:
        y = xy_array[:, 1] - np.mean(xy_array[:, 1])

        ls = LombScargle(x, y, normalization='psd')
        power = ls.power(freqs)

        k_freq = freqs[np.argmin(abs(freqs - k / 2 / np.pi))]
        thetas = ls.model_parameters(k_freq)

        freqs_array[:, 1] = power

    return freqs_array, thetas


def get_spectrum_window(freqs, powers, k, halfwidth=0.001, avg=True):
    idx = (freqs > (k - halfwidth) / 2 / np.pi) & (freqs < (k + halfwidth) / 2 / np.pi)
    not_idx = (freqs <= (k - halfwidth) / 2 / np.pi) | (freqs >= (k + halfwidth) / 2 / np.pi)

    if avg:
        if np.average(powers[not_idx]) == 0.0:
            return 0.0, 0.0
        else:
            return np.average(powers[idx]), np.average(powers[idx]) / np.average(powers[not_idx])
    else:
        return np.sum(powers[idx]), np.average(powers[idx]) * len(idx) / np.average(powers[not_idx])


class KLLoss(nn.Module):
    """
    Custom loss function performing KL loss on soft labels
    """

    def __init__(self, num_classes=10):
        super(KLLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target, label=None):
        if label != None:
            predicted = predicted[label != -100]
            target = target[label != -100]
        num_points = predicted.shape[0]
        num_classes = self.num_classes  # Remove unless binary classification
        cum_losses = predicted.new_zeros(num_points)

        for y in range(num_classes):
            target_temp = predicted.new_full((num_points,), y, dtype=torch.long)
            y_loss = F.cross_entropy(predicted, target_temp, reduction="none")
            cum_losses += target[:, y].float() * y_loss

        return cum_losses.mean()


def generate_plots_nb(xy, freqs, k=30.0):
    x_train_norm = xy[:, 0]
    y_predict_0 = xy[:, 1]
    y_teacher_0 = xy[:, 2]

    fig2, ax = plt.subplots(ncols=2, nrows=1)
    fig2.set_size_inches(12.0, 4.0)
    fig2.subplots_adjust(wspace=0.5)
    ax[0].scatter(x_train_norm, y_teacher_0, marker='o', s=1.5, label='teacher')
    ax[0].scatter(x_train_norm, y_predict_0, marker='o', s=1.5, label='student')
    ax[0].set_xlabel(r'$p$', fontsize=18)
    ax[0].set_ylabel(r'$q_{i^*}$', fontsize=18)
    ax[0].set_ylim([0, 1])
    ax[0].legend(markerscale=4.0)

    y = freqs[:, 1]

    ax[1].scatter(freqs[:, 0] * 2 * np.pi, y, marker='o', s=1.5, label='student', color="orange")
    ax[1].axvline(x=k, ymin=0, ymax=1, linestyle='dotted', color='black', linewidth=2)
    ax[1].set_ylabel(r'$P(f)$', fontsize=18)
    ax[1].set_xlabel(r'$f$', fontsize=18)
    ax[1].set_xscale('log')
    ax[1].legend(markerscale=4.0, loc='lower left')

    fig2.canvas.draw()
    plt.show()

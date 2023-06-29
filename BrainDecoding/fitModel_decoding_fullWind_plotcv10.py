# -*- coding: utf-8 -*-
"""
Created on  20 may 22

@author: degano
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import scipy.stats
from scipy.stats import bootstrap


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m - h, m + h


folderTedLium = 'datafolderRoot'
path = 'PUT_gitPath_of_the_project'

outfile = folderTedLium + '/stimuli_subj/decoding_results_nested_LeftDepProsody_final_classes_windDec_prestimF_10cv.npz'
npzfile = np.load(outfile)

subjS = np.mean(npzfile['subjS'], axis=0)
subjSAll = np.mean(npzfile['subjSAll'], axis=0)

a1 = np.arange(11)
a2 = np.arange(3)
df = pd.DataFrame({'Subject': np.tile(a1, (3,)), 'Classes': np.tile(a2, (11,)),
                   'DecodingAcc': np.reshape(subjSAll, [subjSAll.size, ])})

sns.boxplot(x='Classes', y='DecodingAcc', data=df)
plt.ylim([0.439, 0.603])
# plt.savefig(path+'/images/decodinglogistic_Mvpa_pre.svg', format='svg')

plt.show()

rng = np.random.default_rng()
for i in range(3):
    print(' ')
    # print(mean_confidence_interval(subjSAll[:,i]))
    CIB = bootstrap((subjSAll[:, i],), np.std, confidence_level=0.95,
                    random_state=rng).confidence_interval
    print([np.mean(subjSAll[:, i]), CIB[0], CIB[1]])

# Permutation
STRINGS = ['Max separability', 'inv separability', 'Trained separability']

from mlxtend.evaluate import permutation_test

print(' ')
for i in range(3):
    p_value = permutation_test(subjSAll[:, i], .5 * np.ones((11,)),
                               func='x_mean > y_mean',
                               paired=True,
                               num_rounds=10000,
                               seed=0)
    print(STRINGS[i] + ' pval {0:0f}'.format(p_value))
print(' ')
p_value = permutation_test(subjSAll[:, 0], subjSAll[:, 1],
                           func='x_mean > y_mean',
                           paired=True,
                           num_rounds=10000,
                           seed=0)
print('MAX>INV pval {0:0f}'.format(p_value))

p_value = permutation_test(subjSAll[:, 0], subjSAll[:, 2],
                           func='x_mean > y_mean',
                           paired=True,
                           num_rounds=10000,
                           seed=0)
print('MAX>TRAIN pval {0:0f}'.format(p_value))

p_value = permutation_test(subjSAll[:, 2], subjSAll[:, 1],
                           func='x_mean > y_mean',
                           paired=True,
                           num_rounds=10000,
                           seed=0)
print('TRAIN>INV pval {0:0f}'.format(p_value))

# -*- coding: utf-8 -*-
"""
Created on  03/2023

@author: degano
"""
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy import stats

folderTedLium = 'datafolderRoot'
path = 'PUT_gitPath_of_the_project'

outfile = folderTedLium + 'stimuli_subj/decoding_results_nested_LeftDepProsody_final_classes_logistic_full_slideF_cv10.npz'
npzfile = np.load(outfile)

subjS = np.mean(npzfile['subjS'], axis=0)
Ttimes = npzfile['t']
subjSAll = np.mean(npzfile['subjSAll'], axis=0)
scoreSClasses = np.mean(npzfile['scoreSClasses'], axis=0)

# %%

STRINGS = ['low prosodic strenght and dependency', 'strong prosodic strenght and dependency',
           'low prosodic strenght and no dependency', 'strong prosodic strenght and no dependency ']

for NN in range(4):

    print(STRINGS[NN])
    fig, ax = plt.subplots()
    Meandec = np.mean(scoreSClasses[:, :, NN], axis=0)
    error = 2 * np.std(scoreSClasses[:, :, NN], axis=0) / np.sqrt(scoreSClasses.shape[0])
    ax.plot(Ttimes, np.mean(scoreSClasses[:, :, NN], axis=0), label='score')
    plt.fill_between(Ttimes, Meandec - error, Meandec + error, alpha=.3)
    plt.title(STRINGS[NN])
    ax.axvline(0, color='k', linestyle='--', label='chance')
    ax.axhline(.5, color='k', linestyle='--', label='chance')

    toCluster = scoreSClasses[:, :, NN] - .5
    T_obs, cluster, clusterPval, H0 = mne.stats.permutation_cluster_1samp_test(toCluster, n_permutations=1024, tail=1)
    T_obs_plot = np.nan * np.ones_like(T_obs)
    cnt = 1
    for c, p_val in zip(cluster, clusterPval):
        if p_val <= 0.05:
            print('Cluster {0:0f}'.format(cnt) + ' start {0:3f}'.format(Ttimes[c[0][0]]) +
                  ' stop {0:3f}'.format(Ttimes[c[0][-1]]) + ' pVal {0:0.4f}'.format(p_val))

            T_obs_plot[c] = T_obs[c]
            ax.plot(Ttimes[c], .485 * np.ones(len(c[0])), 'k', linewidth=3)
            cnt = cnt + 1

    plt.ylim([0.45, 0.565])
    # plt.savefig(path+'/images/decodinglogisticCLass'+str(NN)+'.svg', format='svg')
    plt.show()

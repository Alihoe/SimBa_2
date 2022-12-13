import operator

from scipy.stats import spearmanr
import numpy as np


def analyse_correlation(all_features, embeddings, correlation_method, data_name):
        all_feature_names = '_'.join(all_features)
        a = embeddings.shape[0]
        b = embeddings.shape[1]
        c = embeddings.shape[2]
        embeddings = embeddings.reshape(a, b*c)
        if correlation_method == "spearmanr":
            correlation, p_value = spearmanr(embeddings, axis=1, nan_policy='propagate')
            print(correlation)
        with open('../data/evaluation/' + all_feature_names + '_' + data_name + '.txt', 'w') as f:
            print(all_feature_names)
            print(correlation, file=f)


def analyse_feature_correlation(all_features, feature_scores, correlation_method, data_name):
    if correlation_method == "spearmanr":
        correlation, p_value = spearmanr(feature_scores, nan_policy='propagate')

    with open('../data/evaluation/' + data_name + '_correlation.txt', 'w') as f:
        print(all_features, file=f)
        print(correlation, file=f)
        for idx, feature in enumerate(all_features):
            correlations = correlation[idx]
            feature_correlation = dict(zip(all_features, correlations))
            feature_correlation_sorted = dict(sorted(feature_correlation.items(), key=operator.itemgetter(1), reverse=True))
            feature_correlation = dict(list(feature_correlation_sorted.items())[1:])
            print('-------------------------------------------', file=f)
            print('The correlation for feature ' + str(feature), file=f)
            for key, value in feature_correlation.items():
                print('with feature ' + str(key) + ' is ' + str(round(value, 3)), file=f)
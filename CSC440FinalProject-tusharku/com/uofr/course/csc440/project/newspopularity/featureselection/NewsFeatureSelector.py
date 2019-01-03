"""
Class to encapsulate all methods relevant
to selecting the appropriate features in order
to build classifiers that correctly predict the popularity that
an online news article is going to have.
"""

import numpy as np
from skfeature.function.statistical_based import chi_square
from skfeature.function.similarity_based import fisher_score


class NewsFeatureSelector:
    """
    Method to run chi square statistic for relevance between
    features and target values in order to
    find correlation and then rank the features based
    on those correlated value.
    """
    @classmethod
    def rank_features_using_chisquare(cls, data_frame, target_key, cols_to_ignore=None):
        X = data_frame.values
        keys = list(data_frame.keys())
        target_col_idx = keys.index(target_key)

        # Removing the target column from keys
        del keys[target_col_idx]

        # Remove all columns that are asked to be ignored
        if cols_to_ignore is not None:
            for col in cols_to_ignore:
                idx = keys.index(col)
                del keys[idx]

        Y = data_frame.loc[:, target_key].values
        X = data_frame.loc[:, keys]
        neg_test_result = np.any(X < 0, axis=0)
        non_negative_value_columns = [keys[i] for i, res in enumerate(neg_test_result) if not res]

        # Het data for only positive valued columns
        X = data_frame.loc[:, non_negative_value_columns]

        score = chi_square.chi_square(X, Y)
        rank = chi_square.feature_ranking(score)
        ranked_features = [ non_negative_value_columns[i] for i in rank]
        return score, ranked_features, non_negative_value_columns

    """
    Method to run chi square statistic for relevance between
    features and target values in order to
    find correlation and then rank the features based
    on those correlated value.
    """
    @classmethod
    def rank_features_using_fisherscore(cls, data_frame, target_key, cols_to_ignore=None):
        X = data_frame.values
        keys = list(data_frame.keys())
        target_col_idx = keys.index(target_key)

        # Removing the target column from keys
        del keys[target_col_idx]

        # Remove all columns that are asked to be ignored
        if cols_to_ignore is not None:
            for col in cols_to_ignore:
                idx = keys.index(col)
                del keys[idx]

        Y = data_frame.loc[:, target_key].values
        X = data_frame.loc[:, keys]

        score = fisher_score.fisher_score(X, Y)
        rank = fisher_score.feature_ranking(score)
        ranked_features = [keys[i] for i in rank]
        return score, ranked_features, keys

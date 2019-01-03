"""
Class to load the online news data given the file
path where the csv file resides.
This will simply load the dataset and returned a dataframe.
This file does not do any kind of pre-processing work
on the attributes.
"""

import pandas as pd
import numpy as np

ranked_features = ['shares', 'kw_avg_avg', 'LDA_02', 'data_channel_is_world', 'is_weekend',
                       'data_channel_is_entertainment', 'data_channel_is_socmed', 'weekday_is_saturday',
                       'data_channel_is_tech', 'LDA_04', 'kw_min_avg', 'LDA_03', 'weekday_is_sunday', 'num_hrefs',
                       'kw_max_avg', 'LDA_01', 'self_reference_avg_sharess', 'num_imgs', 'global_sentiment_polarity',
                       'global_subjectivity', 'num_keywords', 'LDA_00', 'self_reference_max_shares',
                       'rate_negative_words', 'self_reference_min_shares', 'global_rate_positive_words',
                       'title_sentiment_polarity', 'kw_min_min', 'kw_avg_max', 'rate_positive_words',
                       'abs_title_sentiment_polarity', 'weekday_is_wednesday', 'timedelta', 'n_tokens_title',
                       'title_subjectivity', 'n_tokens_content', 'kw_max_max', 'max_positive_polarity',
                       'data_channel_is_lifestyle','weekday_is_tuesday', 'average_token_length', 'num_self_hrefs',
                       'kw_avg_min','weekday_is_thursday', 'avg_negative_polarity', 'global_rate_negative_words',
                       'data_channel_is_bus', 'kw_max_min', 'avg_positive_polarity', 'num_videos',
                       'min_positive_polarity', 'weekday_is_monday', 'min_negative_polarity', 'kw_min_max',
                       'weekday_is_friday', 'max_negative_polarity', 'n_unique_tokens', 'n_non_stop_words',
                       'n_non_stop_unique_tokens', 'abs_title_subjectivity', 'url']


class NewsDataLoader:

    """
    Main method to load the data using pandas library
    Returns a data frame object
    """
    @classmethod
    def fetch_data(cls, file_path, sep, header, na_filter):
        with open(file_path) as f:
            data_frame = pd.io.parsers.read_csv(file_path, sep=sep, header=header, na_filter=na_filter)
        return data_frame

    """
    Method to format column names of news data frame.
    In particular trim any spaces that the column names
    have and return the data frame with the formatted
    keys set as its column names
    """
    @classmethod
    def format_columns(cls, data_frame):
        strippedKeys = [str.strip(key) for key in data_frame.keys()]
        data_frame.columns = strippedKeys
        return data_frame

    """
    Method to bin the target value of a data frame
    into the desired number of classes to allow
    classification on those binned labels
    """
    @classmethod
    def bin_target_label(cls, target_key, bin_classes, data_frame):
        target_values = data_frame.loc[:, target_key].values
        target_values = sorted(target_values)
        class_max_values = [target_values[int(i * (len(target_values)-1)/bin_classes)]
                            for i in range(1, bin_classes+1)]
        class_min_values = [0] + class_max_values[:-1]
        bin_class_indexes = range(bin_classes)
        for min_val, max_val, bin_idx in zip(class_min_values, class_max_values, bin_class_indexes):
            data_frame[target_key] = data_frame[target_key].mask(
                data_frame[target_key].between(min_val, max_val), bin_idx)


    @classmethod
    def get_data_with_top_k_ranked_features(cls, data_frame, k):
        return data_frame[ranked_features[:k]]

    """
    Method to split the input data frame into
    training validation and test split.
    """
    @classmethod
    def train_validate_test_split(cls, df, train_percent=.6, validate_percent=.2, seed=None):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = df.iloc[perm[:train_end]]
        validate = df.iloc[perm[train_end:validate_end]]
        test = df.iloc[perm[validate_end:]]
        return train, validate, test


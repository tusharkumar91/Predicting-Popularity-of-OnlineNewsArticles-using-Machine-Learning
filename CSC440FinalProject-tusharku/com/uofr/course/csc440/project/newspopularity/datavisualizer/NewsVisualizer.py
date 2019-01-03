"""
CLass to provide all the basic utilities for
visualizing the different data related aspects
of Online News Dataset.
"""

from matplotlib import pyplot as plt


class NewsVisualizer:

    """
    Method to build histogram of given attribute
    in a grid of plots for the provided
    data frame object
    """
    @classmethod
    def plot_histogram_attributes(cls, attribute_keys, data_frame, nrow=1, ncol=1):
        if len(attribute_keys) != (nrow * ncol):
            raise ValueError("length of attributeKeys does not match nrow*ncol")
        f, ax = plt.subplots(nrows=nrow, ncols=ncol, gridspec_kw={"wspace": 0.4, "hspace": 0.7})
        if nrow == 1 and ncol == 1:
            for i, key in enumerate(attribute_keys):
                values = sorted(data_frame.loc[:, key].astype(float))
                values = values[int(0.1 * len(values)):int(0.9 * len(values))]
                ax.hist(values, 100, color=(0.8, 0.827, 0.95), histtype="bar", edgecolor='black',
                        linewidth=0.5)
                #x.set_title("Histogram for {}".format(key.capitalize()))
                ax.set_xlabel(key.capitalize())
                ax.set_ylabel("Frequency")
        elif nrow == 1 or ncol == 1:
            for i, key in enumerate(attribute_keys):
                values = sorted(data_frame.loc[:, key].astype(float))
                values = values[int(0.1 * len(values)):int(0.9 * len(values))]
                ax[i].hist(values, 50, color=(0.8, 0.827, 0.95), histtype="bar", edgecolor='black', linewidth=0.5)
                ax[i].set_title("Histogram for {}".format(key))
        else:
            key_idx = 0
            for row in range(nrow):
                for col in range(ncol):
                    values = sorted(data_frame.loc[:, attribute_keys[key_idx]].astype(float))
                    values = values[int(0.1 * len(values)):int(0.9 * len(values))]
                    ax[row][col].hist(values, 50,  color=(0.8, 0.827, 0.95), histtype="bar", edgecolor='black', linewidth=0.5)
                    ax[row][col].set_title("Histogram for {}".format(attribute_keys[key_idx]), fontsize=10)
                    key_idx += 1
        plt.show()

    @classmethod
    def plot_feature_scores(cls, feature_scores, feature_keys, method):
        plt.barh(feature_keys, feature_scores, alpha=0.5, edgecolor='black', linewidth=1, align="center", left=-1)
        plt.xlabel('{0} Score for attribute'.format(method), fontsize=10)
        plt.yticks(fontsize=4)
        #plt.xlim(-1, 18)
        plt.xticks([])
        #plt.title('Plot of {0} score between attribute and target label values'.format(method), fontsize=8)
        plt.show()
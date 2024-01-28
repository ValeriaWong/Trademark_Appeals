import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

class MiScores:
    def __init__(self, data):
        self.data = data

    def make_mi_scores(self, X, y, discrete_features):
        mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores

    def plot_mi_scores(self, scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    def compute_and_plot_mi(self, selected_features, target_column, discrete_features):
        X = self.data[selected_features]
        y = self.data[target_column]

        # Encode the categorical columns
        for colname in X.select_dtypes("object"):
            X.loc[:, colname] = X[colname].factorize()[0]

        mi_scores = self.make_mi_scores(X, y, discrete_features)
        plt.figure(dpi=100, figsize=(10, 6))
        self.plot_mi_scores(mi_scores)
        plt.show()
        return mi_scores

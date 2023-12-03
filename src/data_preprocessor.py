import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data, selected_features):
        self.data = data
        self.selected_features = selected_features
        self.vectorizer = TfidfVectorizer(max_features=100)

    def convert_to_tfidf_and_add(self, column_name):
        column_values = self.data[column_name].fillna("").astype(str)
        tfidf_matrix = self.vectorizer.fit_transform(column_values)
        tfidf_array = tfidf_matrix.toarray()
        for i in range(tfidf_array.shape[1]):
            self.data[f"{column_name}_tfidf_{i}"] = tfidf_array[:, i]

    def preprocess(self):
        for column in self.selected_features:
            self.convert_to_tfidf_and_add(column)
        return self.data

    def split_data(self, target_column, test_size=0.3, valid_size=0.5):
        X = self.data.drop(columns=[target_column])
        Y = self.data[target_column].map({'予以驳回': 0, '予以初步审定': 1, '部分驳回': 2})
        X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=test_size, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from typing import List

__all__ = ['rfClassifier']


class rfClassifier:
    def __init__(self, config: dict):
        self.max_depth = config["max_depth"]
        self.n_estimators = config["n_estimators"]
        self.n_folds = config["n_folds"]
        self.features = config["features"]
        self.col_label = config["col_label"]
        self.clf = None

    def run_cross_validation(self, df_train: pd.DataFrame) -> List[float]:
        """
        performs cross validation and return cv f1 scores
        :param df_train: pd.DataFrame
        :param col_label: str, label column
        :return: List[float]
        """
        clf = RandomForestClassifier(
            max_depth=self.max_depth,
            random_state=0,
            n_estimators=self.n_estimators
        )

        X_train = df_train[self.features].values
        y_train = df_train[self.col_label].values
        return cross_val_score(clf, X_train, y_train, cv=self.n_folds, scoring="f1", verbose=1)

    def fit_rf_classifier(self, df_train: pd.DataFrame) -> None:
        clf = RandomForestClassifier(
            max_depth=self.max_depth,
            random_state=0,
            n_estimators=self.n_estimators
        )

        X_train = df_train[self.features].values
        y_train = df_train[self.col_label].values
        clf.fit(X_train, y_train)
        self.clf = clf

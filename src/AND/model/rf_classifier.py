from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
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
        self.cv_scores = None
        self.clf = None
        self.importances = None

    def run_cross_validation(self, df_train: pd.DataFrame) -> dict:
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
        self.cv_scores = cross_validate(clf, X_train, y_train, cv=self.n_folds, scoring=["f1", "precision", "recall"],
                                        verbose=1, return_train_score=True)
        return self.cv_scores

    def fit_classifier(self, df_train: pd.DataFrame) -> dict:
        clf = RandomForestClassifier(
            max_depth=self.max_depth,
            random_state=0,
            n_estimators=self.n_estimators
        )

        X_train = df_train[self.features].values
        y_train = df_train[self.col_label].values
        clf.fit(X_train, y_train)
        self.clf = clf
        self.importances = clf.feature_importances_
        return self.importances

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].values
        predictions = self.clf.predict(X)
        df["prediction"] = predictions
        return df

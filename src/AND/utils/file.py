import os
from pathlib import Path
from typing import List
import pandas as pd
import AND.utils.logger as logger
import pickle

__all__ = ['FileUtil']


class FileUtil:

    def __init__(self, config: dict):
        prefix = Path(os.path.abspath(os.path.realpath(__file__))).parents[3]
        self.data_path = os.path.join(prefix, config["data"]["data"])
        self.gt_path = os.path.join(prefix, config["data"]["ground_truth"])
        self.persons_path = os.path.join(prefix, config["data"]["persons"])
        self.results_training_path = os.path.join(prefix, config["results"]["training"])
        self.results_test_path = os.path.join(prefix, config["results"]["test"])
        self.config = config

    def read_data(self) -> List[pd.DataFrame]:
        """
        Loads the data into data frames
        :return: List[pd.DataFrame]
        """
        data = pd.read_json(self.data_path)
        gt = pd.read_json(self.gt_path)
        persons = pd.read_json(self.persons_path)
        logger.logging.info(">>> Loaded data successfully")
        return [data, gt, persons]

    def report_cv_scores(self, cv_scores: dict) -> None:
        result_object = dict(
            config=self.config,
            cv_scores=cv_scores
        )

        file = self.results_training_path + 'cv_scores.pkl'
        with open(file, 'wb') as handle:
            pickle.dump(result_object, handle)

    def report_feature_importances(self, feature_importances: dict) -> None:
        result_object = dict(
            config=self.config,
            importances=feature_importances
        )

        file = self.results_training_path + 'feature_importances.pkl'
        with open(file, 'wb') as handle:
            pickle.dump(result_object, handle)

    def report_profiles(self, profiles: List[set], df_test: pd.DataFrame) -> None:
        result_object = dict(
            config=self.config,
            profiles=profiles,
            testpersonid=list(df_test["personId"].unique())
        )

        file = self.results_test_path + 'author_profiles.pkl'
        with open(file, 'wb') as handle:
            pickle.dump(result_object, handle)

    def report_test_results(self, mean_purity: float, mean_fragmentation: float, scores: dict):
        result_object = dict(
            config=self.config,
            mean_purity=mean_purity,
            mean_fragmentation=mean_fragmentation,
            classificatin_scores=scores
        )

        file = self.results_test_path + 'scores.pkl'
        with open(file, 'wb') as handle:
            pickle.dump(result_object, handle)

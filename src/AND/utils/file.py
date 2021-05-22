import os
from pathlib import Path
from typing import List
import pandas as pd
import AND.utils.logger as logger


__all__ = ['FileUtil']


class FileUtil:

    def __init__(self, config: dict):
        prefix = Path(os.path.abspath(os.path.realpath(__file__))).parents[3]
        self.data_path = os.path.join(prefix, config["data"]["data"])
        self.gt_path = os.path.join(prefix, config["data"]["ground_truth"])
        self.persons_path = os.path.join(prefix, config["data"]["persons"])

    def read_data(self) -> List[pd.DataFrame]:
        """
        Loads the data into data frames
        :return: List[pd.DataFrame]
        """
        data = pd.read_json(self.data_path)
        gt = pd.read_json(self.gt_path)
        persons = pd.read_json(self.persons_path)
        logger.logging.info(
            "Loaded data successfully"
        )
        return [data, gt, persons]

import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, List
from sklearn.metrics import classification_report

__all__ = ['evaluate_profiles', 'estimate_classification_scores']


def evaluate_profiles(profiles: List[set], df_gt: pd.DataFrame) -> Tuple[float]:
    """
    Computes purity and fragmentation of the profiles
    :param profiles: lsit[set]
    :param df_gt: pd.DataFrame
    :return: List[
    """
    mapping = {}
    for contributionId, personsId in df_gt[["contributionId", "personId"]].values:
        mapping[contributionId] = personsId

    purities = []
    for profile in profiles["profiles"]:
        personIds = [mapping[contribution] for contribution in profile]
        c = Counter(personIds)
        purities.append(c.most_common(1)[0][1] / len(personIds))

    mean_purity = np.mean(purities)

    personId_list = []
    for profile in profiles["profiles"]:
        personIds = set([mapping[contribution] for contribution in profile])
        personId_list.extend(personIds)

    c = Counter(personId_list)
    mean_fragmentation = np.mean(list(c.values()))
    return mean_purity, mean_fragmentation

def estimate_classification_scores(df:pd.DataFrame) -> dict:
    y_true = df["same_person"].values
    y_pred = df["prediction"].values
    results = classification_report(y_true, y_pred, output_dict= True)
    return results["macro avg"]
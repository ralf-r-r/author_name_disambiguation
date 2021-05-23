import pandas as pd
from typing import List
import random
import numpy as np
import AND.utils.logger as logger

__all__ = ['combine_data_sets', 'create_train_test', 'create_contribution_pairs']


def combine_data_sets(df_contr: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    combines ground truth and contributions data
    :param df_contr: contributions
    :param df_gt:
    :return:
    """
    gt_persons = {k: v for (k, v) in df_gt[["contributionId", "personId"]].values}
    df_contr["personId"] = df_contr["contribution_id"].map(gt_persons)
    return df_contr


def create_train_test(df: pd.DataFrame, persons: List[str], ratio: float) -> List[pd.DataFrame]:
    """
    makes a train test split of the data set such that x % of the personId's are in the training set
    and 1-x % are in the test set
    :param df: pd.DataFrame , the trnaing data set
    :param persons: List[str], list of person ID's
    :param ratio: float, ratio of the train test split
    :return: List[pd.DataFrrame], the train and test data sets
    """
    random.Random(4).shuffle(persons)
    tranining_persons = persons[0:int(len(persons) * ratio)]
    test_persons = persons[int(len(persons) * ratio):]
    df_train = df[df["personId"].isin(tranining_persons)]
    df_test = df[df["personId"].isin(test_persons)]
    return [df_train, df_test]


def create_contribution_pairs(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    creates pairs of contributions
    :param df: pd.DataFrame
    :param n: int, only take every nt-h negative pair into the traning set
    :return df_pairs:  pd.DataFrame
    """
    columns = ['contribution_id',
               'first_name_cleaned',
               'middle_name_cleaned',
               'last_name_cleaned',
               'full_name_cleaned',
               'workplace_cleaned',
               'workplace_locations',
               'focus_areas_cleaned',
               'orgs_cleaned',
               'gpes_cleaned',
               'personId']

    columns_part_2 = [c + "_2nd" for c in columns]

    array = df[columns].values
    data_dict = {}

    count = 0
    for k in range(0, array.shape[0]):
        if k % 1000 == 0:
            logger.logging.info(">>> Created pairs for " + str(k) + " of " + str(array.shape[0]) + " contributions")
        for m in range(k, array.shape[0]):
            values1 = array[k]
            values2 = array[m]
            if values1[10] == values2[10]:
                values = np.concatenate([values1, values2], axis=0)
                data_dict["row_" + str(k) + "_" + str(m)] = values
            else:
                count += 1
                if count % n == 0:
                    values = np.concatenate([values1, values2], axis=0)
                    data_dict["row_" + str(k) + "_" + str(m)] = values

    columns.extend(columns_part_2)
    df_pairs = pd.DataFrame.from_dict(data_dict, orient='index', columns=columns)
    df_pairs["same_person"] = df_pairs["personId_2nd"] == df_pairs["personId"]
    df_pairs["same_person"] = df_pairs["same_person"].astype(int)
    return df_pairs

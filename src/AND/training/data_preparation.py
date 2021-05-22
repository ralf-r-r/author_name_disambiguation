import pandas as pd
from typing import List
import random


__all__ = ['combine_data_sets', 'create_train_test']


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


def create_train_test(df: pd.DataFrame, persons: List[str]) -> List[pd.DataFrame]:
    """
    makes a 80:20 train test split, such that 80 % of the authors are in the
    training data set and 20% of authors are in the test data set
    :param df: pd.DataFrame , the trnaing data set
    :param persons: List[str], list of person ID's
    :return: List[pd.DataFrrame], the train and test data sets
    """
    random.shuffle(persons)
    tranining_persons = persons[0:int(len(persons)*0.7)]
    test_persons = persons[int(len(persons)*0.7):]
    df_train = df[df["personId"].isin(tranining_persons)]
    df_test = df[df["personId"].isin(test_persons)]
    return [df_train, df_test]


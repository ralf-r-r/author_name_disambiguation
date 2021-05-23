import pandas as pd
import mpu
import numpy as np
import jellyfish
from typing import List, Tuple
import AND.utils.logger as logger

__all__ = ['compute_features']


def exact_match(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    Check whether two author names are an exact match
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return: pd.DataFrame
    """
    df[feature_name] = df[col1] == df[col2]
    df[feature_name] = df[feature_name].astype(int)

    return df


def soundex_string_matching(s1: str, s2: str) -> int:
    try:
        if (len(s1) == 0) or (len(s2) == 0):
            return 0
        soundex1 = jellyfish.soundex(s1)
        soundex2 = jellyfish.soundex(s2)
        return int(soundex1 == soundex2)
    except UnicodeDecodeError:
        return 0



def soundex(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    compute phonetics based exact matching with soundex algorithm
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return: pd.DataFrame
    """
    df[feature_name] = df.apply(lambda x: soundex_string_matching(x[col1], x[col2]), axis=1)
    return df


def shared_strings(l1: List[str], l2: List[str]) -> int:
    a = set(l1)
    b = set(l2)
    return len(a & b)


def number_shared_in_list(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    compute the number of shared strings in two lists
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return: pd.DataFrame
    """
    df[feature_name] = df.apply(lambda x: shared_strings(x[col1], x[col2]), axis=1)
    return df


def haversine_distance_work_locations(l1: List[List[float]], l2: List[List[float]]) -> Tuple[int,int,int]:
    if len(l1) == 0 or len(l2) == 0:
        return 99999,99999,99999
    else:
        distances = []
        for t1 in l1:
            for t2 in l2:
                distances.append(int(mpu.haversine_distance(t1, t2) / 1000))

        return min(distances), max(distances), int(np.mean(distances))


def distance_average_work_locations(df: pd.DataFrame, col1: str, col2: str, feature_names: List[str]) -> pd.DataFrame:
    """
    computes the distance between the averaged work locations of two authors
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_names: List[str], the column names of the computed features
    :return: pd.DataFrame
    """
    df[feature_names] = df.apply(lambda x: haversine_distance_work_locations(x[col1], x[col2]), axis=1,  result_type="expand")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    compute multiple features for the similarity classification task
    :param df:
    :return: pd.DataFrame
    """

    columns = ['first_name_cleaned',
               'middle_name_cleaned',
               'last_name_cleaned',
               'full_name_cleaned',
               "workplace_cleaned"]

    logger.logging.info(">>> Computing exact matches")
    for col in columns:
        df = exact_match(df, col1=col, col2=col + "_2nd", feature_name="exact_match_" + col)

    logger.logging.info(">>> Computing soundex exact matches")
    columns.remove("full_name_cleaned")

    for col in columns:
        df = soundex(df, col1=col, col2=col + "_2nd", feature_name="soundex_" + col)

    logger.logging.info(">>> Computing number of shared words")
    df = number_shared_in_list(df,
                               col1="focus_areas_cleaned",
                               col2="focus_areas_cleaned_2nd",
                               feature_name="no_shared_focus_area")
    df = number_shared_in_list(df,
                               col1="gpes_cleaned",
                               col2="gpes_cleaned_2nd",
                               feature_name="no_shared_gpes")
    df = number_shared_in_list(df,
                               col1="orgs_cleaned",
                               col2="orgs_cleaned_2nd",
                               feature_name="no_shared_orgs")

    logger.logging.info(">>> Computing distances between work locations")
    dist_features = ["min_distance_km", "max_distance_km", "mean_distance_km"]
    df = distance_average_work_locations(df,
                                         col1="workplace_locations",
                                         col2="workplace_locations_2nd",
                                         feature_names=dist_features)

    return df

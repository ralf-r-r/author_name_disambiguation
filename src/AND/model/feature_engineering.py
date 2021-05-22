import pandas as pd
from fuzzywuzzy import fuzz
import jellyfish
from typing import List

__all__ = ['compute_features']

def exact_match(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    Check whether two author names are an exact match
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return:
    """
    df[feature_name] = df[col1] == df[col2]
    df[feature_name] = df[feature_name].astype(int)

    return df


def levenshtein_v1(s1: str, s2: str) -> int:
    if (len(s1) == 0) or (len(s2) == 0):
        return 0
    return fuzz.ratio(s1, s2)


def levenshtein_v2(s1: str, s2: str) -> int:
    if (len(s1) == 0) or (len(s2) == 0):
        return 0
    return fuzz.partial_ratio(s1, s2)


def levenshtein(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    compute string similarities using Levenshtein Distance
    https://github.com/seatgeek/fuzzywuzzy
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return:
    """

    df[feature_name] = df.apply(lambda x: levenshtein_v1(x[col1], x[col2]), axis=1)
    df[feature_name + "partial"] = df.apply(lambda x: levenshtein_v2(x[col1], x[col2]), axis=1)
    return df


def soundex_string_matching(s1: str, s2: str) -> int:
    try:
        if (len(s1) == 0) or (len(s2) == 0):
            return 0
        soundex1 = jellyfish.soundex(s1)
        soundex2 = jellyfish.soundex(s2)
    except UnicodeDecodeError:
        return 0
    return int(soundex1 == soundex2)


def soundex(df: pd.DataFrame, col1: str, col2: str, feature_name: str) -> pd.DataFrame:
    """
    compute phonetics based exact matching with soundex algorithm
    :param df: pd.DataFrame, dataset with pairwise contributions
    :param col1: str, column name of contribution1
    :param col2: str,  column name of contribution2
    :param feature_name: str, the column name of the computed feature
    :return:
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
    :return:
    """
    df[feature_name] = df.apply(lambda x: shared_strings(x[col1], x[col2]), axis=1)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    compute multiple features for the similarity classification task
    :param df:
    :return:
    """

    name_columns = ['first_name_cleaned',
                    'middle_name_cleaned',
                    'last_name_cleaned',
                    'full_name_cleaned',
                    "workplace_cleaned"]

    print("computing exact matches")
    for col in name_columns:
        df = exact_match(df, col1=col, col2=col + "_2nd", feature_name="exact_match" + col)

    print("computing soundex exact matches")
    name_columns = ['first_name_cleaned',
                    'middle_name_cleaned',
                    'last_name_cleaned']
    for col in name_columns:
        df = soundex(df, col1=col, col2=col + "_2nd", feature_name="soundex" + col)

    print("computing number of  shared wordss")
    df = number_shared_in_list(df, col1="focus_areas_cleaned", col2="focus_areas_cleaned_2nd", feature_name="no_shared_focus_area")
    df = number_shared_in_list(df, col1="gpes_cleaned", col2="gpes_cleaned_2nd", feature_name="no_shared_gpes")
    df = number_shared_in_list(df, col1="orgs_cleaned", col2="orgs_cleaned_2nd", feature_name="no_shared_orgs")

    return df

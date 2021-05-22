import pandas as pd
import re
from typing import List

__all__ = ['cleaning_procedure']


def clean_name(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    cleans a column of type str
    :param df: pd.DataFrame
    :param col: str, name of the column
    :return: pd.DataFrame
    """
    df[col + "_cleaned"] = df[col].str.replace(".", "", regex=True)
    df[col + "_cleaned"] = df[col + "_cleaned"].str.replace("-", " ", regex=True)
    df[col + "_cleaned"] = df[col + "_cleaned"].str.lower()
    df[col + "_cleaned"] = df[col + "_cleaned"].str.replace(' +', ' ', regex=True)
    return df


def clean_list_of_strings(l: List[str]) -> List[str]:
    """
    cleans a list of strings
    :param l: List[str]
    :return: List[str]
    """
    if l:
        for k, s in enumerate(l):
            s = s.lower()
            s = s.replace('.', ' ')
            s = s.replace('/', ' ')
            s = s.replace('-', ' ')
            s = s.replace(',', ' ')
            s = s.replace(';', ' ')
            s = s.replace(':', ' ')
            s = s.replace('[', ' ')
            s = s.replace(']', ' ')
            s = s.replace('(', ' ')
            s = s.replace(')', ' ')
            s = re.sub(' +', ' ', s)
            l[k] = s
        return l
    else:
        return []


def clean_str_list(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    cleans a column of type List[str]
    :param df: pd.DataFrame
    :param col: str, name of the column
    :return: pd.DataFrame
    """
    df[col + "_cleaned"] = df[col].apply(clean_list_of_strings)
    return df


def cleaning_procedure(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleans multiple columns of the data set
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    df = clean_name(df, col="full_name")
    df = clean_name(df, col="middle_name")
    df = clean_name(df, col="last_name")
    df = clean_name(df, col="first_name")
    df = clean_name(df, col="workplace")

    df = clean_str_list(df, col="focus_areas")
    df = clean_str_list(df, col="gpes")
    df = clean_str_list(df, col="orgs")

    droplist = ["full_name",
                "middle_name",
                "last_name",
                "first_name",
                "workplace",
                "focus_areas",
                "gpes",
                "orgs"]

    df = df.drop(droplist, axis=1)
    return df

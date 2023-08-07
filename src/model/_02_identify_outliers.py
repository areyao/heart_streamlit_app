# Databricks notebook source
from src.utils._run_scripts import *
from src.model._run_scripts import *
from src.utils.utilities import *

def handle_outliers(dataframe, filt_col, threshold):
    """
    handle_outliers : Identifies and removes outliers in a dataframe

    :param dataframe: The dataframe to be assessed
    :param filt_col: A list of columns to identify outliers
    :param threshold : the threshold of the iqr
    :return dataframe: The dataframe with outliers removed
    """
    for df_col in filt_col:
        # Calculates the first and third quartile (To define the IQR)
        q1 = dataframe.approxQuantile(df_col, [0.25], 0.0)[0]
        q3 = dataframe.approxQuantile(df_col, [0.75], 0.0)[0]
        iqr = q3 - q1

        # Change threshold to parameter
        # Upper and lower bound for acceptable values
        lower_bound = q1 - iqr * threshold
        upper_bound = q3 + iqr * threshold

        # Redefines the dataframe to only include rows that aren't outliers
        dataframe = dataframe.filter((f.col(df_col) >= lower_bound) & (f.col(df_col) <= upper_bound))

    return dataframe

# Databricks notebook source
from src.utils._run_scripts import *
from src.model._run_scripts import *

def count_null(dataframe):
    """
    count_null : Counts null values if there are any in the dataset

    :param dataframe: the dataframe to assess null values from

    :return null_values: the count null values from the dataframe
    """

    null_values = dataframe.select(
        [
            f.count(f.when(f.isnull(col), col)).alias(col)
            for col in dataframe.columns
        ]
    )
    return null_values

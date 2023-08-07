# Databricks notebook source
from src.utils._run_scripts import *
from src.model._run_scripts import *
def get_mdn_or_md(dataframe, stat_type):
    """
    get_mdn_or_md : Retrieves the median or mode of the columns of the passed dataframe

    :param dataframe: The dataframe to be assessed
    :param stat_type: The type of desc. stat to return. ("mode" or "median")
    :return stat_list: A list of the dataframe's corresponding column statistic values.
    
    :raises ValueError: If a value other than "mode" or "median" is passed as an arg.
    """
    # Will hold the list of the dataframe's corresponding column's statistic values
    stat_list = [stat_type]

    # Will calculate the mode or median depending on what arg is passed
    if stat_type == "mode":
        summary_list = [dataframe.groupby(col).count()
                        .orderBy("count", ascending=False).first()[0]
                        for col in dataframe.columns]
    elif stat_type == "median":
        summary_list = [dataframe.groupBy()
                        .agg(f.percentile_approx(col, 0.5)).first()[0]
                        for col in dataframe.columns]
    else:
        raise ValueError("Invalid argument 'type' - Only accepted values : 'median', 'mode'")

    # Will append all the calculated statistic values into the stat_list
    for val in summary_list:
        stat_list.append(val)

    return stat_list

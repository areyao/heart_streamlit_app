from src.utils._run_scripts import *
from src.model._run_scripts import *

def get_age_group(dataframe):
    """
        get_age_group : Retrieves the age group of the entry and appends a new column called age_group

        :param dataframe: The dataframe to be assessed
        :return dataframe: A dataframe with age_group of entries classified and appended as a new column.
        """
    dataframe = dataframe.withColumn(
        "age_group",
        f.when((f.col("Age") >= 25) & (f.col("Age") <= 29), "late 20s")
        .when((f.col("Age") >= 30) & (f.col("Age") <= 34), "early 30s")
        .when((f.col("Age") >= 35) & (f.col("Age") <= 39), "late 30s")
        .when((f.col("Age") >= 40) & (f.col("Age") <= 44), "early 40s")
        .when((f.col("Age") >= 45) & (f.col("Age") <= 49), "late 40s")
        .when((f.col("Age") >= 50) & (f.col("Age") <= 54), "early 50s")
        .when((f.col("Age") >= 55) & (f.col("Age") <= 59), "late 50s")
        .when((f.col("Age") >= 60) & (f.col("Age") <= 64), "early 60s")
        .when((f.col("Age") >= 65) & (f.col("Age") <= 69), "late 60s")
        .when((f.col("Age") >= 70) & (f.col("Age") <= 74), "early 70s")
        .when((f.col("Age") >= 75) & (f.col("Age") <= 79), "late 70s"),
    )
    return dataframe

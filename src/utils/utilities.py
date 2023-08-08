from src.utils._run_scripts import *

# General Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as f
from pyspark.sql.types import *

# Pipeline Libraries
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, UnivariateFeatureSelector
from pyspark.ml import Pipeline, PipelineModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier, \
    LinearSVC, RandomForestClassificationModel, LogisticRegressionModel
from pyspark.ml.functions import vector_to_array

# Evaluation and Tuning Libraries
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

# Streamlit App Libraries
import streamlit as st
import datetime as dt
import shap
import plotly.express as px

# Environment Libraries
from pyspark.sql import SparkSession
import os
import sys
from pyspark.sql import SparkSession
from pyspark import SQLContext
# os.environ['SPARK_HOME'] = "C:\spark-3.4.1-bin-hadoop3"
os.environ['SPARK_HOME'] = "C:\spark-3.4.1-bin-hadoop3"
# os.environ['JAVA_HOME'] = "C:\Program Files\Eclipse Adoptium\jdk-11.0.15.10-hotspot"
os.environ['JAVA_HOME_1'] = "C:\Program Files\Java\jdk1.8.0_181"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# spark = SparkSession.builder.config("spark.sql.execution.arrow.pyspark.enabled", False).config("spark.driver.host","localhost").getOrCreate()
spark = SparkSession.builder.getOrCreate()
conf = get_conf()

def read_csv(path):
    """
    Reads the path and takes the csv within to process.

    Parameters:
        path (string) : file path of the .csv file

    Returns:
        dataframe (spark dataframe) : the csv file as a spark dataframe with an inferred schema
    """
    dataframe = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path)
    return dataframe

def calculateAge(birthDate):
    today = dt.datetime.today()
    age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
    return age


def predict(preprocessor, base_model, meta_preprocessor, meta_model, row):
    heart_data = read_csv('data/heart.csv')
    # heart_data = heart_data = get_heart_info(conf)
    df_schema = heart_data.drop('HeartDisease').schema
    to_predict = spark.createDataFrame(row, df_schema)

    # Preprocess and run user input into pipeline and base model
    heart_data = preprocessor.transform(to_predict)
    base_pred = base_model.transform(heart_data)

    # Select only relevant features to get final prediction from
    meta_features_df = base_pred.select("pred_rf", "prob_rf")

    # Vectorize the relevant features
    meta_features_transformed = meta_preprocessor.transform(meta_features_df)

    # Make final prediction
    meta_pred = meta_model.transform(meta_features_transformed)

    value = meta_pred.select('meta_prediction').collect()[0][0]

    if (int(value) == 1):
        st.error(f'You are at risk for Heart Disease. Please consult a medical professional.')
    else:
        st.success(f'You are not at risk for Heart Disease.')
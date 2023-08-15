# Databricks notebook source
def get_conf():
    # Returns conf : a value based on dictionary key
    conf = {
        "paths": {
            "heart_data_path": "../data/heart.csv",
            "heart_data_st_path": "data/heart.csv"
        },
        "required_columns": {
            "categorical_columns": ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
            "continuous_columns": ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
            # "categorical_columns": ['FastingBS', 'RestingECG', 'ExerciseAngina'],
            # "continuous_columns": ['Age', 'RestingBP', 'Oldpeak'],
            "label_column": "HeartDisease",
            "heart_data_columns": ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                                   'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
                                   'HeartDisease'],
            "transformed_columns" : ['Sex_index', 'ChestPainType_index', 'FastingBS_index', 'RestingECG_index', 'ExerciseAngina_index', 'ST_Slope_index','Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        },
        "meta_columns": {
            "meta_features" : ["pred_rf"],
                # , "pred_lr", "pred_lsvc", "pred_gbt"],
            "meta_cont_features" : ["prob_rf"],
            #"prob_lr"
            "meta_label_col" : "label",
            "meta_predCol" : "meta_prediction",
            "meta_labelCol" : "meta_label",
            "meta_featuresCol" : "meta_features",
        },
        "iqr_thresh": {"threshold": 1.5},
        "train_val_test_split": [0.8, 0.1, 0.1],
        # "train_val_test_split": [0.8, 0., 0.3],
        "summary_type": ["mode", "median"]
    }

    return conf

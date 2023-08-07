# from src.utils._run_scripts import *
# from src.model._run_scripts import *
#
# def final_model_pipeline():
#     stages = []
#     preprocessor = PipelineModel.load('../../pipeline/preprocessor')
#     stages.append(preprocessor)
#
#     base_model = RandomForestClassificationModel.load('../../pipeline/base_model')
#     stages.append(base_model)
#     base_model_pred = base_model.select("pred_rf", "prob_rf")
#     stages.append(base_model_pred)
#
#     meta_preprocessor = PipelineModel.load('../../pipeline/meta_preprocessor')
#     meta_preprocessor_pred = meta_preprocessor.transform(base_model_pred)
#     stages.append(meta_preprocessor_pred)
#
#     meta_model = LogisticRegressionModel.load('../../pipeline/meta_model')
#     stages.append(meta_model)
#
#     final_pipeline = Pipeline(stages=stages)
#
#     return final_pipeline
#
# conf = get_conf()
# heart_data = get_heart_info(conf,True,"../../data/heart.csv")
# input_data = [[40, 'M', 'ATA', 140, 289, 0, 'Normal', 172, 'N', 0.0, 'Up']]
# heart_data_schema = heart_data.drop('HeartDisease').schema
#
# # student_dataframe
#
# input_data_df = spark.createDataFrame(input_data, heart_data_schema)
# model_pipeline = final_model_pipeline()
# final_pred = model_pipeline.transform(input_data_df)
# final_pred.show()
# # heart_data = preprocessor.transform(input_data_df)
# # base_pred = base_model.transform(heart_data)
# #
# # # Select only relevant features to get final prediction from
# # meta_features_df = base_pred.select("pred_rf", "prob_rf")
# #
# # meta_features_transformed = meta_preprocessor.transform(meta_features_df)
# #
# # meta_prediction = meta_model.transform(meta_features_transformed)
# #
# # meta_prediction.show()
#
# # +-------+--------------------+------------+--------------------+--------------------+--------------------+--------------------+---------------+
# # |pred_rf|             prob_rf|bin_features|       cont_features|       meta_features|       rawPrediction|         probability|meta_prediction|
# # +-------+--------------------+------------+--------------------+--------------------+--------------------+--------------------+---------------+
# # |    0.0|[0.94536547182002...|       [0.0]|[0.94536547182002...|[0.0,0.9453654718...|[3.25147447782282...|[0.96272605980843...|            0.0|
# # +-------+--------------------+------------+--------------------+--------------------+--------------------+--------------------+---------------+
# #
#
# # Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
#

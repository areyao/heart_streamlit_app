from src.utils._run_scripts import *
from src.model._run_scripts import *

preprocessor = PipelineModel.load('./preprocessor')
base_model = RandomForestClassificationModel.load('./base_model')
meta_preprocessor = PipelineModel.load('./meta_preprocessor')
meta_model = LogisticRegressionModel.load('./meta_model')

conf = get_conf()
heart_data = get_heart_info(conf)
input_data = [[40,'M','ATA',140,289,0,'Normal',172,'N',0.0,'Up']]
heart_data_schema = heart_data.drop('HeartDisease').schema

# student_dataframe

input_data_df = spark.createDataFrame(input_data,heart_data_schema)
heart_data = preprocessor.transform(input_data_df)
base_pred = base_model.transform(heart_data)

# Select only relevant features to get final prediction from
meta_features_df = base_pred.select("pred_rf", "prob_rf")

meta_features_transformed = meta_preprocessor.transform(meta_features_df)

meta_prediction = meta_model.transform(meta_features_transformed)

meta_prediction.show()

# Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease
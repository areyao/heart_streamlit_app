from src.utils._run_scripts import *
from src.model._run_scripts import *

preprocessor = PipelineModel.load('./preprocessor')

conf = get_conf()
heart_data = get_heart_info(conf, handleOutliers=True)
heart_data = preprocessor.transform(heart_data)

rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                  predictionCol='pred_rf', probabilityCol='prob_rf')

base_model = rf.fit(heart_data)

# base_model.save('./base_model')
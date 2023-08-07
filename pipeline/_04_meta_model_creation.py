from src.utils._run_scripts import *
from src.model._run_scripts import *

preprocessor = PipelineModel.load('./preprocessor')
base_model = RandomForestClassificationModel.load('./base_model')
meta_preprocessor = PipelineModel.load('./meta_preprocessor')

conf = get_conf()
heart_data = get_heart_info(conf)
heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])
heart_data = preprocessor.transform(heart_data)
base_pred = base_model.transform(heart_data)

# Select only relevant features to get final prediction from
meta_features_df = base_pred.select("pred_rf", "prob_rf", "label")

meta_features_df = meta_preprocessor.transform(meta_features_df)

meta_predCol="meta_prediction"
meta_labelCol="meta_label"
meta_featuresCol = "meta_features"

# instantiate the classifier and the parameter field
meta_classifier = LogisticRegression(featuresCol=meta_featuresCol,
                             labelCol=meta_labelCol,
                             predictionCol=meta_predCol)

meta_param = tune.ParamGridBuilder() \
        .addGrid(meta_classifier.regParam, [0.0, 0.01, 0.1]) \
        .addGrid(meta_classifier.elasticNetParam, [0.0, 0.01, 0.1]) \
        .addGrid(meta_classifier.maxIter, [50, 100, 200]) \
        .build()

meta_lr = grid_search_model(meta_classifier, meta_param)

meta_model = meta_lr.fit(meta_features_df).bestModel

meta_model_transformed = meta_model.transform(meta_features_df)

meta_model_transformed.show()

# meta_model.save('./meta_model')
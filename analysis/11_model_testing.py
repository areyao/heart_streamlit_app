from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)
heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'],
                             conf['iqr_thresh']['threshold'])
train_df, validation_df, test_df = split_data(heart_data,  [0.7,0.0,0.3])

lr = LogisticRegression(featuresCol='features', labelCol='label',
                        predictionCol='pred_lr', probabilityCol='prob_lr',
                        rawPredictionCol='rawPred_lr')

rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                            predictionCol='pred_rf', probabilityCol='prob_rf')

# bestModel_pipe = build_data_pipeline(False)
# bestModel_train_df = bestModel_pipe.fit(train_df).transform(train_df)

# Extract Best Model
# lr_cv = base_model_cv(lr)
# lr_bestModel = extract_best_model(lr_cv, bestModel_train_df)
# rf_cv = base_model_cv(rf)
# rf_bestModel = extract_best_model(rf_cv, bestModel_train_df)

models = [rf]

# Contains the random forest model
base_pipeline = build_data_pipeline(True, models)

base_pipeline_model = base_pipeline.fit(train_df)

base_pred = base_pipeline_model.transform(train_df)

# Select base model features to pass into the meta model
meta_features_df = base_pred.select("pred_rf", "prob_rf", "label")

# -------------------------------- META CLASSIFIER ---------------------------------------- +

meta_predCol = "meta_prediction"
meta_labelCol = "meta_label"
meta_featuresCol = "meta_features"

# instantiate the classifier
meta_classifier = LogisticRegression(featuresCol=meta_featuresCol,
                                     labelCol=meta_labelCol,
                                     predictionCol=meta_predCol, probabilityCol = "prb_lr")
# meta_classifier = RandomForestClassifier(featuresCol=meta_featuresCol,
#                              labelCol=meta_labelCol,
#                              predictionCol=meta_predCol)

# build meta pipeline
meta_pipeline = build_meta_pipeline(True)
meta_pipe = meta_pipeline.fit(meta_features_df)

meta_features_df = meta_pipe.transform(meta_features_df)

# create the parameters list
meta_param = tune.ParamGridBuilder() \
    .addGrid(meta_classifier.regParam, [0.0, 0.01, 0.1]) \
    .addGrid(meta_classifier.elasticNetParam, [0.0, 0.01, 0.1]) \
    .addGrid(meta_classifier.maxIter, [50, 100, 200]) \
    .build()
# meta_param = tune.ParamGridBuilder()\
#     .addGrid(meta_classifier.maxDepth, [2, 5, 10, 20, 30])\
#     .addGrid(meta_classifier.maxBins, [10, 20, 40, 80, 100])\
#     .addGrid(meta_classifier.numTrees, [10, 20, 30])\
#     .build()

# Build the grid search model and obtain the best model
meta_lr = grid_search_model(meta_classifier, meta_param)
meta_model = meta_lr.fit(meta_features_df).bestModel
# meta_model = meta_lr.fit(meta_features_df)

# -------------------------------- Test all models with the test set --------------------------------------
test_pred = base_pipeline_model.transform(test_df)
meta_test_df = test_pred.select("pred_rf", "prob_rf", "label")

# meta classifier predictions on the test set's meta features
meta_test_trans = meta_pipe.transform(meta_test_df)
meta_test_pred = meta_model.transform(meta_test_trans)

meta_test_pred.show()

# choose an evaluator
evaluator = evals.BinaryClassificationEvaluator()
evaluator.setLabelCol(meta_labelCol)

# calculate roc and pr scores
roc_meta = evaluator.evaluate(meta_test_pred, {evaluator.metricName: "areaUnderROC"})
pr_meta = evaluator.evaluate(meta_test_pred, {evaluator.metricName: "areaUnderPR"})

binary_eval = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# record the confusion matrix metrics on the test set
accuracy, precision, recall, f1, binaryEval, curveMetrics = evaluate_metrics(meta_test_pred, 'meta_prediction', binary_eval)
print(f"Accuracy : {accuracy}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")
print(f"F1 : {f1}")
print(f"AUC-ROC : {binaryEval}")

# Visualize ROC Curve
# plt.subplots(1, figsize=(10,10))
# x_val = [x[0] for x in curveMetrics]
# y_val = [x[1] for x in curveMetrics]
# plt.title('Receiver Operating Characteristic')
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.plot(x_val, y_val)
# plt.show()
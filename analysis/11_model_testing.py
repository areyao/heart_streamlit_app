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
accuracy, precision, recall, f1, binaryEval, tp, fp, tn, fn, curveMetrics = evaluate_metrics(meta_test_pred, 'meta_prediction', binary_eval)
print(accuracy)
print(precision)
print(recall)
print(f1)
print(binaryEval)


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
# rf then lr (actual bestModel) - 7-0-3
# Accuracy = 0.9264705882352942
# Precision = 0.9166666666666666
# Recall = 0.9428571428571428
# f1 = 0.9295774647887323
# AUC-ROC Score = 0.9584415584415584


# rf-bestModel then lr-normalModel (7-0-3)
# 0.9095022624434389
# 0.9008264462809917
# 0.9316239316239316
# 0.9159663865546218
# 0.9517587113740961

# rf - bestModel then lr - bestModel (7-0-3)
# 0.8914027149321267
# 0.8604651162790697
# 0.9487179487179487
# 0.9024390243902439
# 0.9580867850098622

# rf - normal model then lr - bestModel (7-0-3)
# 0.8959276018099548
# 0.8852459016393442
# 0.9230769230769231
# 0.9037656903765692
# 0.9572649572649573

# rf - normalModel then lr - normalModel (7-0-3)
# 0.8823529411764706
# 0.8582677165354331
# 0.9316239316239316
# 0.8934426229508197
# 0.9495397764628536


# LR
# lr, gbt
# lr
# 0.8529411764705882
# 0.8048780487804879
# 0.9428571428571428
# 0.868421052631579
# 0.9558441558441559


# LR , LR(best model)
# accuracy 0.9117647058823529
# 0.8918918918918919
# 0.9428571428571428
# 0.9166666666666667
# 0.9463203463203465

# rf then lr (bestModel) --- IDEAL
# 0.9264705882352942
# 0.9166666666666666
# 0.9428571428571428
# 0.9295774647887323
# 0.961038961038961

# rf then rf(bestModel)
# 0.8823529411764706
# 0.8461538461538461
# 0.9428571428571428
# 0.8918918918918919
# 0.929004329004329

# lr then rf(bestModel)
# 0.9117647058823529
# 0.8918918918918919
# 0.9428571428571428
# 0.9166666666666667
# 0.9380952380952381

# rf(bestModel), rf(bestModel)
# 0.8382352941176471
# 0.875
# 0.8
# 0.8358208955223881
# 0.8787878787878789

# rf(bestModel), lr(bestModel) - train_df used (can be discarded)
# 0.8823529411764706
# 0.8648648648648649
# 0.9142857142857143
# 0.888888888888889
# 0.9523809523809523

# rf(bestModel), lr(bestModel) - 8-1-1 - w/second scaling
# 0.9264705882352942
# 0.9166666666666666
# 0.9428571428571428
# 0.9295774647887323
# 0.9593073593073593

# rf then lr (bestModel) - w/second scaling - 7-0-3train_df used
# 0.8959276018099548
# 0.873015873015873
# 0.9401709401709402
# 0.9053497942386831
# 0.9578402366863905

# rf then lr (bestModel) - 8-1-1 removed second scaling
# 0.8823529411764706
# 0.8461538461538461
# 0.9428571428571428
# 0.8918918918918919
# 0.9523809523809523

# rf then lr (bestModel) - 7-0-3 removed second scaling
# 0.9004524886877828
# 0.88
# 0.9401709401709402
# 0.909090909090909
# 0.9523339907955296

# rf then lr (actual bestModel) - 8-1-1 removed second scaling
# 0.8823529411764706
# 0.8461538461538461
# 0.9428571428571428
# 0.8918918918918919
# 0.948051948051948



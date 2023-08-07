# from src.utils._run_scripts import *
# from src.model._run_scripts import *
#
# conf = get_conf()
# heart_data = get_heart_info(conf)
# heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])
# heart_data = heart_data.select('FastingBS', 'RestingECG', 'ExerciseAngina', 'Age', 'RestingBP', 'Oldpeak', 'HeartDisease')
#
# train_df, validation_df, test_df = split_data(heart_data)
#
# # Obtain prediction of validation class from the base model pipeline
# base_model = PipelineModel.load('./base_model_pl_corr')
# base_prediction = base_model.transform(test_df)
#
# # # create the meta features dataset
# # meta_features_df = base_prediction.select("pred_lr","pred_lsvc",
# #                                     "prob_lr", "pred_gbt",
# #                                     "label")
# #
# # meta_predCol="meta_prediction"
# # meta_labelCol="meta_label"
# # meta_featuresCol = "meta_features"
# #
# # # instantiate the classifier
# # meta_classifier = LogisticRegression(featuresCol=meta_featuresCol,
# #                              labelCol=meta_labelCol,
# #                              predictionCol=meta_predCol)
# #
# # # build specific pipeline
# # meta_pipeline = build_meta_pipeline(meta_classifier)
# #
# # # create the parameters list
# # meta_param = tune.ParamGridBuilder() \
# #         .addGrid(meta_classifier.regParam, [0.0, 0.01, 0.1]) \
# #         .addGrid(meta_classifier.elasticNetParam, [0.0, 0.01, 0.1]) \
# #         .addGrid(meta_classifier.maxIter, [50, 100, 200]) \
# #         .build()
# #
# # # build the grid search model
# # meta_lr = grid_search_model(meta_pipeline, meta_param)
# #
# # # train the model
# # meta_model = meta_lr.fit(meta_features_df)
# #
# # # base classifiers make predictions on the test set
# # test_pred = base_model.transform(test_df)
# #
# # # create the meta features test dataset
# # meta_test_df = test_pred.select("pred_lr","pred_lsvc",
# #                                     "prob_lr","pred_gbt",
# #                                     "label")
# #
# # # meta classifier predictions on the meta features test set
# # meta_test_pred = meta_model.transform(meta_test_df)
#
# binary_evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
# accuracy, precision, recall, f1, binaryEval = evaluate_metrics(base_prediction, 'prediction', binary_evaluator)
# print(accuracy)
# print(precision)
# print(recall)
# print(f1)
# print(binaryEval)
#
# # meta_model.save('./meta_model_pl_vec')

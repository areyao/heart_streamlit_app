# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#
# from src.utils._run_scripts import *
# from src.model._run_scripts import *
# from pyspark.ml.classification import MultilayerPerceptronClassifier
#
# conf = get_conf()
# heart_data = get_heart_info(conf)
# heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])
# # heart_data = heart_data.select('FastingBS', 'RestingECG', 'ExerciseAngina', 'Age', 'RestingBP', 'Oldpeak', 'HeartDisease')
#
# # Only perform vector assembly for features
# bestModel_pipeline = build_data_pipeline(False)
#
# # Reformat heart data to include features
# rf_heart_data = bestModel_pipeline.fit(heart_data).transform(heart_data)
#
# # Split data into train, validation, and testing
# # model_train, model_validation, model_test = split_data(rf_heart_data)
# train_df, test_df = rf_heart_data.randomSplit([0.6,0.4],1)
#
# # Base models to use for stacking method
# lr = LogisticRegression(featuresCol='features', labelCol='label',
#                         predictionCol='pred_lr', probabilityCol='prob_lr',
#                         rawPredictionCol='rawPred_lr')
#
# lsvc = LinearSVC(featuresCol='features', labelCol='label',
#                 predictionCol='pred_lsvc', rawPredictionCol='rawPred_lsvc')
#
# gbt = GBTClassifier(featuresCol = 'features', labelCol="label", predictionCol='pred_gbt')
#
# # dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
#
# layers = [4,5,5,3]
# mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features",layers = layers, seed = 1)
#
# # Obtain best model for each base model
# # Steps :
# # 1. Obtain CrossValidator for each base model
# # lr_cv = base_model_cv(lr)
# # lsvc_cv = base_model_cv(lsvc)
# # gbt_cv = base_model_cv(gbt)
# # df_cv = base_model_cv(dt)
# # 2. Obtain bestModel when fitted with training data
# # lr_bestModel = extract_best_model(lr_cv, model_train)
# # lsvc_bestModel = extract_best_model(lsvc_cv, model_train)
# # gbt_bestModel = extract_best_model(gbt_cv, model_train)
# # df_bestModel = extract_best_model(df_cv, model_train)
#
# # Best Models to Stack
# # models = [lsvc_bestModel, lr_bestModel, gbt_bestModel]
#
# # models = [mlp]
#
# # Split original data
# # train_df, validation_df, test_df = split_data(heart_data)
#
# # Append best models into the base model pipeline
# # base_pipeline = build_data_pipeline(True,models)
#
# # Fit pipeline onto training data set
# # base_model = base_pipeline.fit(train_df)
# base_model = mlp.fit(train_df)
# #
# base_test_pred = base_model.transform(test_df)
#
#
# # base_test_pred.show()
# # evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
# # mlpacc = evaluator.evaluate(base_test_pred)
#
#
# # results = testResults.select(['prediction', 'label']).withColumn('prediction',col('prediction').cast('integer'))
# results = base_test_pred.withColumn('label', f.col('label').cast('double')).select(['prediction', 'label'])
# predictionAndLabels = results.rdd
# metrics = MulticlassMetrics(predictionAndLabels)
#
# cm = metrics.confusionMatrix().toArray()
# accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
# precision = (cm[0][0]) / (cm[0][0] + cm[1][0])
# recall = (cm[0][0]) / (cm[0][0] + cm[0][1])
# f1 = 2 * ((precision * recall) / (precision + recall))
#
#
# # print("MLP ACCURACY", mlpacc)
#
# # binary_evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
# # binROC = binary_evaluator.evaluate(base_test_pred)
# # print("BE ROC", binROC)
#
# # binary_evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
# # accuracy, precision, recall, f1, binaryEval = evaluate_metrics(base_test_pred, 'prediction', binary_evaluator)
# # print(accuracy)
# # print(precision)
# # print(recall)
# # print(f1)
# # print(binaryEval)
#
# # base_model.save('./base_model_pl_vec')
#
# #GBT
# # 0.8051948051948052
# # 0.8416666666666667
# # 0.7952755905511811
# # 0.8178137651821863
# # 0.8655360387643852
#
# # LR
# # 0.7705627705627706
# # 0.8303571428571429
# # 0.7322834645669292
# # 0.7782426778242679
# # 0.8615233192004853
#
# #SVM
# # 0.7835497835497836
# # 0.8290598290598291
# # 0.7637795275590551
# # 0.7950819672131147
# # 0.8644760751059972
#
# #SVM W 4 FEATURES
# # 0.7835497835497836
# # 0.8598130841121495
# # 0.7244094488188977
# # 0.7863247863247863
# # 0.8631889763779531
#
# #GBT W 4 FEAUTRESS
# # 0.8008658008658008
# # 0.8292682926829268
# # 0.8031496062992126
# # 0.816
# # 0.8608419139915203
#
# # GBT w vector assembler
# # 0.8008658008658008
# # 0.8292682926829268
# # 0.8031496062992126
# # 0.816
# # 0.8658010296789823
#
# #DT
# #only 70's

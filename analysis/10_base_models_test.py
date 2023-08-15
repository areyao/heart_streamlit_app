from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf, handleOutliers=True)

# Build the pipeline to preprocess the data for the feature vectorization
bestModel_pipe = build_data_pipeline(appendModels=False)
heart_data = bestModel_pipe.fit(heart_data).transform(heart_data)

train_df, validation_df, test_df = split_data(heart_data, [0.7,3.0,0.0])

lr = LogisticRegression(featuresCol='features', labelCol='label')

rf = RandomForestClassifier(featuresCol='features', labelCol='label')

gbt = GBTClassifier(featuresCol='features', labelCol='label')

lsvc = LinearSVC(featuresCol='features', labelCol='label')

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')

model_titles = ["LogisticRegression", "RandomForest", "GradientBoost","SVC","DecisionTree"]
model_preds = []

lr_model = lr.fit(train_df)
lr_val_pred = lr_model.transform(validation_df)
model_preds.append(lr_val_pred)

rf_model = rf.fit(train_df)
rf_val_pred = rf_model.transform(validation_df)
model_preds.append(rf_val_pred)

gbt_model = gbt.fit(train_df)
gbt_val_pred = gbt_model.transform(validation_df)
model_preds.append(gbt_val_pred)

lsvc_model = lsvc.fit(train_df)
lsvc_val_pred = lsvc_model.transform(validation_df)
model_preds.append(lsvc_val_pred)

dt_model = dt.fit(train_df)
dt_val_pred = dt_model.transform(validation_df)
model_preds.append(dt_val_pred)

binary_evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
prediction_summary = []
for num in range(len(model_titles)):
    accuracy, precision, recall, f1, binaryEval = evaluate_metrics(model_preds[num], 'prediction', binary_evaluator)
    prediction_summary.append([model_titles[num], float(round(accuracy*100,2)), float(round(precision*100,2)), float(round(recall*100,2)), float(round(f1*100,2)), float(round(binaryEval*100,2))])

print("-----Base Models Evaluation-----")
summary_schema = f.StructType([
    StructField("Model",StringType(),True),
    StructField("Accuracy",FloatType(),True),
    StructField("Precision",FloatType(),True),
    StructField("Recall",FloatType(),True),
    StructField("F1",FloatType(),True),
    StructField("ROC",FloatType(),True)
])
model_summary = spark.createDataFrame(prediction_summary, summary_schema)
model_summary.show()

# Best Models with ParamGrid and CV
best_models_preds = []

lr_cv = base_model_cv(lr)
lr_bestModel = extract_best_model(lr_cv, train_df)
lr_bestModel_pred = lr_bestModel.transform(validation_df)
best_models_preds.append(lr_bestModel_pred)

rf_cv = base_model_cv(rf)
rf_bestModel = extract_best_model(rf_cv, train_df)
rf_bestModel_pred = rf_bestModel.transform(validation_df)
best_models_preds.append(rf_bestModel_pred)

gbt_cv = base_model_cv(gbt)
gbt_bestModel = extract_best_model(gbt_cv, train_df)
gbt_bestModel_pred = gbt_bestModel.transform(validation_df)
best_models_preds.append(gbt_bestModel_pred)

lsvc_cv = base_model_cv(lsvc)
lsvc_bestModel = extract_best_model(lsvc_cv, train_df)
lsvc_bestModel_pred = lsvc_bestModel.transform(validation_df)
best_models_preds.append(lsvc_bestModel_pred)

dt_cv = base_model_cv(dt)
dt_bestModel = extract_best_model(dt_cv, train_df)
dt_bestModel_pred = dt_bestModel.transform(validation_df)
best_models_preds.append(dt_bestModel_pred)

bm_prediction_summary = []
for num in range(len(model_titles)):
    accuracy, precision, recall, f1, binaryEval, curveMetrics = evaluate_metrics(best_models_preds[num], 'prediction', binary_evaluator)
    bm_prediction_summary.append([model_titles[num], float(round(accuracy*100,2)), float(round(precision*100,2)), float(round(recall*100,2)), float(round(f1*100,2)), float(round(binaryEval*100,2))])

print("-----Best Models Evaluation-----")
best_model_summary = spark.createDataFrame(bm_prediction_summary, summary_schema)
best_model_summary.show()

#0.7,0.0,0.3
# # -----Base Models Evaluation-----
# +------------------+--------+---------+------+-----+-----+
# |             Model|Accuracy|Precision|Recall|   F1|  ROC|
# +------------------+--------+---------+------+-----+-----+
# |LogisticRegression|   84.14|    82.26| 89.67|85.81|91.91|
# |      RandomForest|   85.03|    85.29|  87.0|86.14|91.79|
# |     GradientBoost|   78.79|    77.18| 85.67| 81.2|86.59|
# |               SVC|   85.38|    85.39| 87.67|86.51|91.69|
# |      DecisionTree|   81.46|    82.03| 83.67|82.84|84.09|
# +------------------+--------+---------+------+-----+-----+
# -----Best Models Evaluation-----
# +------------------+--------+---------+------+-----+-----+
# |             Model|Accuracy|Precision|Recall|   F1|  ROC|
# +------------------+--------+---------+------+-----+-----+
# |LogisticRegression|   83.42|    81.08|  90.0|85.31|91.87|
# |      RandomForest|   84.67|    84.74|  87.0|85.86|92.14|
# |     GradientBoost|   85.74|    85.26| 88.67|86.93|91.65|
# |               SVC|   84.49|    83.18|  89.0|85.99|91.77|
# |      DecisionTree|   83.07|    81.35| 88.67|84.85|79.27|
# +------------------+--------+---------+------+-----+-----+

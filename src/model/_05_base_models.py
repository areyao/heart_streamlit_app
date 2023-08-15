from src.utils._run_scripts import *
from src.model._run_scripts import *

def base_model_cv(baseModel):
    """
    base_mode_cv : Retrieves a CrossValidator based on the passed base model.

    :param baseModel (pyspark.ml.classification) : a classification model
    :return crossVal : CrossValidator for the model

    :raises ValueError: If a value other than a classification model is passed
    """
    evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
    grid = tune.ParamGridBuilder()
    crossVal = tune.CrossValidator()

    # Only accept classification models
    acceptable_base_models = ['LinearSVC', 'LogisticRegression', 'GBTClassifier', 'DecisionTreeClassifier', "RandomForestClassifier"]
    # Retrieves the model name within the function
    base_model_name = type(baseModel).__name__

    if base_model_name in acceptable_base_models:
        if base_model_name == acceptable_base_models[0]:  # LinearSVC / SVM
            # evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC', rawPredictionCol='rawPred_lsvc')
            evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
            grid = tune.ParamGridBuilder() \
                .addGrid(baseModel.regParam, np.arange(0., .1, .01)) \
                .addGrid(baseModel.maxIter, [5, 10]) \
                .build()
            crossVal = tune.CrossValidator(estimator=baseModel, evaluator=evaluator, estimatorParamMaps=grid)
            return crossVal
        elif base_model_name == acceptable_base_models[1]:  # LogisticRegression
            # evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC', rawPredictionCol='rawPred_lr')
            evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')
            grid = tune.ParamGridBuilder().addGrid(baseModel.regParam, np.arange(0., .1, .01)).addGrid(baseModel.elasticNetParam, [0., 1.]).build()
            crossVal = tune.CrossValidator(estimator=baseModel, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
            return crossVal
        elif base_model_name == acceptable_base_models[2] or base_model_name == acceptable_base_models[3]: #GBT
            grid = tune.ParamGridBuilder().addGrid(baseModel.maxDepth, [2, 5, 10, 20, 30]).addGrid(baseModel.maxBins, [10, 20, 40, 80, 100]).build()
            crossVal = tune.CrossValidator(estimator=baseModel, evaluator=evaluator, estimatorParamMaps=grid)
            return crossVal
        else: # DT & RF
            grid = tune.ParamGridBuilder().addGrid(baseModel.maxDepth, [2, 5, 10, 20, 30]).addGrid(baseModel.maxBins, [10, 20, 40, 80, 100]).addGrid(baseModel.numTrees, [10, 20, 30]).build()
            crossVal = tune.CrossValidator(estimator=baseModel, estimatorParamMaps=grid, evaluator=evaluator,
                                           numFolds=5)
            return crossVal
    else:
        raise ValueError(
            "Invalid argument 'baseModel' - Only accepted classification models : 'LinearSVC', 'LogisticRegression', 'GBTClassifier', 'DecisionTreeClassifier', RandomForestClassifier'"
        )

from src.utils._run_scripts import *
from src.model._run_scripts import *

def evaluate_metrics(testResults, predCol, evaluatorType):
    results = testResults.withColumn('label', f.col('label').cast('double')).select([predCol, 'label'])
    predictionAndLabels = results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)

    cm = metrics.confusionMatrix().toArray()
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    precision = (cm[0][0]) / (cm[0][0] + cm[1][0])
    recall = (cm[0][0]) / (cm[0][0] + cm[0][1])
    f1 = 2 * ((precision * recall) / (precision + recall))
    binaryEval = evaluatorType.evaluate(testResults)
    return accuracy, precision, recall, f1, binaryEval
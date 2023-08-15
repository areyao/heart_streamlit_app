from src.utils._run_scripts import *
from src.model._run_scripts import *


class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter,
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2,
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)


def evaluate_metrics(testResults, predCol, evaluatorType):
    '''
    evaluate_metrics : retrieves various metrics for a binary classification problem

    :param testResults (pySpark dataframe) : results after testing
    :param predCol (string) : name of the prediction column
    :param evaluatorType (string) : type of evaluator used (ROC)

    :returns accuracy, precision, recall, f1, binaryEval, points (double, double, double, double, double, list) :
        various binary evaluation metrics
    '''

    # Retrieves confusion matrix metrics using the label and the predictions
    results = testResults.withColumn('label', f.col('label').cast('double')).select([predCol, 'label'])
    predictionAndLabels = results.rdd
    metrics = MulticlassMetrics(predictionAndLabels)

    cm = metrics.confusionMatrix().toArray()

    # Calculates the accuracy, precision, recall, f1, and the evaluator score
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    precision = (cm[0][0]) / (cm[0][0] + cm[1][0])
    recall = (cm[0][0]) / (cm[0][0] + cm[0][1])
    f1 = 2 * ((precision * recall) / (precision + recall))
    binaryEval = evaluatorType.evaluate(testResults)

    # get curve points for ROC visualization
    preds = testResults.select('label', 'prb_lr').rdd.map(
        lambda row: (float(row['prb_lr'][1]), float(row['label'])))
    points = CurveMetrics(preds).get_curve('roc')

    return accuracy, precision, recall, f1, binaryEval, points
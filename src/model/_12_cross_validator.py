from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
meta_label = conf['meta_columns']['meta_label_col']

def grid_search_model(pipeline, paramGrid):
    """
    Creates a cross validation object and performs grid search
    over a set of parameters.

    INPUT:
        param = grid of parameters
        pipeline = model pipeline

    OUTPUT:
        cv = cross validation object
    """

    # choose an evaluator
    evaluator = evals.BinaryClassificationEvaluator()
    evaluator.setLabelCol(meta_label)

    # create the cross-validation object
    cv = tune.CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5,
                        parallelism=2)
    return cv
from src.utils._run_scripts import *
from src.model._run_scripts import *
def extract_best_model (crossVal, trainData):
    """
    extract_best_mode : extracts the best model from using the CrossValidator

    Parameters:
        crossVal (CrossValidator) : crossValidator to fit the training data

    Returns:
        best_model (ML model) : the best model to from the crossValidator
    """
    models = crossVal.fit(trainData)
    best_model = models.bestModel
    return best_model
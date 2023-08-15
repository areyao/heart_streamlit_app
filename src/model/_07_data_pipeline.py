from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
categorical_columns = conf['required_columns']['categorical_columns']
continuous_columns = conf['required_columns']['continuous_columns']
label = conf['required_columns']['label_column']


def build_data_pipeline(appendModels,models=[]):
    """
    build_data_pipeline : Combines all the stages of the data processing.

    :param appendModels (boolean) : if models to be appended [essential to distinguish pipeline for best model fitting]
    :param models (list) : list of models to be appended

    :returns pipeline (pipeline) : pipeline to fit training data
    """
    # stages in the pipeline
    stages = []

    # encode the labels
    label_indexer = StringIndexer(inputCol=label, outputCol="label")
    stages.append(label_indexer)

    # iterate through all categorical values
    for categoricalCol in categorical_columns:
        # create a string indexer for each categorical value and assign a new name including the word 'Index'
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + '_index')
        # append the string Indexer to our list of stages
        stages.append(stringIndexer)

    indexed_columns = [colm + "_index" for colm in categorical_columns]

    for contCol in continuous_columns:
        indexed_columns.append(contCol)

    vec_assembler = VectorAssembler(inputCols = indexed_columns, outputCol ='vec_features')
    stages.append(vec_assembler)

    # normalize the continuous features
    scaler = StandardScaler(inputCol="vec_features", outputCol="features"
                            ,withStd=True, withMean=True)
    stages.append(scaler)

    if appendModels:
        stages += models

    # create a pipeline
    pipeline = Pipeline(stages=stages)

    return pipeline

from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
meta_features = conf['meta_columns']['meta_features']
meta_cont_features = conf['meta_columns']['meta_cont_features']
meta_label = conf['meta_columns']['meta_label_col']

def build_meta_pipeline(withLabel):
    """
    Combines all the stages of the meta features processing.
    """
    # stages in the pipeline
    stages = []

    # encode the labels
    if withLabel:
        label_indexer = StringIndexer(inputCol=meta_label, outputCol="meta_label")
        stages.append(label_indexer)

    # encode the binary features
    bin_assembler = VectorAssembler(inputCols=meta_features, outputCol="bin_features")
    stages.append(bin_assembler)

    # encode the continuous features
    cont_assembler = VectorAssembler(inputCols=meta_cont_features, outputCol="cont_features")
    stages.append(cont_assembler)

    # pass all to the vector assembler to create a single sparse vector
    all_assembler = VectorAssembler(inputCols=["bin_features", "cont_features"],
                                    outputCol="meta_features")
    stages.append(all_assembler)

    pipeline = Pipeline(stages=stages)

    return pipeline
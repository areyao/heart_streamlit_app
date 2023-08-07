from src.utils._run_scripts import *
from src.model._run_scripts import *

preprocessor = PipelineModel.load('./preprocessor')
base_model = RandomForestClassificationModel.load('./base_model')

conf = get_conf()
heart_data = get_heart_info(conf)
heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])
heart_data = preprocessor.transform(heart_data)

base_pred = base_model.transform(heart_data)

# Select only relevant features to get final prediction from
meta_features_df = base_pred.select("pred_rf", "prob_rf", "label")

# Create a pipeline to convert the base predictions into features
meta_processor_pipe = build_meta_pipeline(True)

meta_pipe = meta_processor_pipe.fit(meta_features_df)

# meta_pipe_transformed = meta_pipe.transform(meta_features_df)

# meta_pipe.save('./meta_preprocessor')
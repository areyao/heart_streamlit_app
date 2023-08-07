from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)
heart_data = handle_outliers(heart_data, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])

preprocess_pl = build_data_pipeline(False)

preprocess_pipe = preprocess_pl.fit(heart_data)

preprocess_transform_pipe = preprocess_pipe.transform(heart_data)

# preprocess_pipe.save('./preprocessor')
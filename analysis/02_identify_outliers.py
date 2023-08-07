from src.utils._run_scripts import *
from src.model._run_scripts import *

"""
02_identify_outliers:
    This notebook identifies any outliers and from the numeric columns,
    and removes any if there are. It prices the remaining amount of entries left within the data
    after outlier handling.
"""

conf = get_conf()

heart_data = get_heart_info(conf)

threshold = conf["iqr_thresh"]['threshold']
filt_cols = conf["required_columns"]['continuous_columns']
outliers_heart_data = handle_outliers(heart_data,filt_cols, threshold)

print("Number of entries BEFORE outlier handling: ", heart_data.count())
print("Number of entries AFTER outlier handling: ", outliers_heart_data.count())
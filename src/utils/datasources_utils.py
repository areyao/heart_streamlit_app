from src.model._02_identify_outliers import handle_outliers
from src.utils._run_scripts import *
def get_heart_info(conf, handleOutliers = False, streamlitPath = False):
    """
    Returns cab information
    :param: conf (dict): Contains configurations
    :return: pyspark dataframe: cab_data_info
    """
    if streamlitPath:
        heart_data_info = read_csv(conf["paths"]["heart_data_st_path"])
    else:
        heart_data_info = read_csv(conf["paths"]["heart_data_path"])

    if handleOutliers:
        heart_data_info = handle_outliers(heart_data_info, conf['required_columns']['continuous_columns'], conf['iqr_thresh']['threshold'])
    return heart_data_info

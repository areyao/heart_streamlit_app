from src.utils._run_scripts import *
from src.model._run_scripts import *

"""
01_identify_null_values:
    This notebook identifies any null values, if they exist, and counts them.
"""

conf = get_conf()

heart_data = get_heart_info(conf)

count_null(heart_data).show()
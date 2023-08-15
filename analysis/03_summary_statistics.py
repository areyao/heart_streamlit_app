from src.utils._run_scripts import *
from src.model._run_scripts import *

"""
03_summary_statistics:

    This notebook retrieves the summary statistic of the dataset.
"""

conf = get_conf()

heart_data = get_heart_info(conf)

# Returns the summary statistics of the heart_data dataset
summary_stat = heart_data.summary("count", "stddev", "min", "max", "25%", "50%", "75%", "mean")

# Obtain the median
med_list = [get_mdn_or_md(heart_data, stat_type = 'median')]
median_row = spark.createDataFrame(med_list, summary_stat.schema)
summary_stat = summary_stat.union(median_row)

# Obtain the mode
md_list = [get_mdn_or_md(heart_data, stat_type = "mode")]
mode_row = spark.createDataFrame(md_list, summary_stat.schema)
summary_stat = summary_stat.union(mode_row)

summary_stat.show()

"""
From the summary data, we can infer that :
    - the mean age surveyed was 54 years of age, with the youngest being 28, and the oldest being 77.
    - most entries were from male responders
    - most of the responders had no heart diseases
"""
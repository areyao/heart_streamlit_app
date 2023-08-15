from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)

"""
06_diabetes:

    This notebook delves into the fasting blood pressure feature that focuses on the presence of diabetes in an entry to look
    briefly into it's possible relation with heart disease.
"""

"""
To assess this hypothesis, we must look into the fbs, or the Fasting Blood Sugar level.
According to the CDC, a fasting blood sugar level of 99 mg/dL or lower is normal, 100 to 125 mg/dL indicates you have prediabetes, and 126 mg/dL or higher indicates you have diabetes.

However, in the case of our dataset which has already been processed, we can assume that 0 means no Diabetes, and 1 is with Diabetes.
"""

by_diab = heart_data.filter(f.col('HeartDiabetes') == 1).groupBy('FastingBS').agg(f.count(f.col('HeartDisease')).alias('instances'))
by_diab.show()

"""
Most of the instances of heart disease occur to those without diabetes.
"""
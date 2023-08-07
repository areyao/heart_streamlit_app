from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)

"""
To better understand cholesterol levels, we can classfy them according to the National Heart, Lung, and Blood Institute guidelines.
- A total cholesterol level of less than 200 mg/dL (5.17 mmol/L) is normal.

- A total cholesterol level of 200 to 239 mg/dL (5.17 to 6.18 mmol/L) is borderline high.

- A total cholesterol level of 240 mg/dL (6.21 mmol/L) or greater is high.
"""

by_chol = heart_data.withColumn('chol_category', f.when(f.col('Cholesterol') < 200, 'normal').f.when((f.col('Cholesterol') >= 200) & (f.col('Cholesterol') <= 239), 'borderline_high').otherwise('high'))
by_chol.select('Cholesterol', 'chol_category').show()
chol_instances = by_chol.filter(f.col('output') == 1).groupBy('chol_category').agg(f.count(f.col('output')).alias('instances'))
chol_instances.show()

"""
As expected, we can see that most of those at risk have high cholesterol. However, we can also see that there is are those with normal cholesterol levels that are at risk for heart disease.
"""
"""
We can compare these findings with blood pressure (systolic). The following categories were recommended by the American Heart Association.
(Systolic mm Hg (upper number))
- NORMAL - LESS THAN 120
- ELEVATED	120 – 129
- HIGH BLOOD PRESSURE (HYPERTENSION) STAGE 1 - 130 – 139
- HIGH BLOOD PRESSURE (HYPERTENSION) STAGE 2 - 140 OR HIGHER
- HYPERTENSIVE CRISIS - HIGHER THAN 180
"""

by_bp = by_chol.withColumn(
    "bp_category",
    f.when(f.col("trtbps") < 120, "Normal")
    .when((f.col("trtbps") >= 120) & (f.col("trtbps") <= 129), "Elevated")
    .when((f.col("trtbps") >= 130) & (f.col("trtbps") <= 139), "Hypertension Stage 1")
    .when((f.col("trtbps") >= 140) & (f.col("trtbps") <= 180), "Hypertension Stage 2").when((f.col("trtbps") > 180), "Hypertensive Crisis")
)
chol_and_bp = by_bp.filter(f.col('output') == 1).groupBy('bp_category', 'chol_category').agg(f.count(f.col('output')).alias('instances')).orderBy('instances', ascending = False)
chol_and_bp.show()

"""
We can see that regardless of bp_category, chol_category takes most of the instances of heart disease. There are also come cases that even with normal bp and normal cholesterol levels, the risk of heart disease may be present.
"""
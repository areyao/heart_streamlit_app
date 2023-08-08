from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)

"""
We can further categorize this by classifying them within the 5 year age group classification to easily identify 
which groups are more prone to heart failures.
"""

by_5_year_age = getAgeGroup(heart_data)
five_year_df = by_5_year_age.filter(f.col('HeartDisease') == 1).groupBy('age_group', 'HeartDisease').agg(f.count('HeartDisease').alias('instances'))
five_year_df.show()

five_year_df.toPandas().sort_values(by=["instances"],ascending = False).pivot(index = "HeartDisease", columns="age_group", values="instances").plot(kind='bar', title = "Instances per Age Group")
plt.show()
"""
Through this classification, we can see that those with the most instances of Heart Risk are in their 40s, 50s, and 60s.
We can also see that is only one instance of heart risk in their early 20's and late 70's.
"""

"""
Apart from the age, we can also look at which gender is more prominent to heart risk.
"""
by_gender = heart_data.filter(f.col('HeartDisease') == 1).groupBy('Sex').agg(f.count(f.col('HeartDisease')).alias('instances'))
by_gender.show() # Make Visualizer

"""
We can see males have more instances of being at risk than females.
We can look at this alongside the age range to understand how the distributions changes by age.
"""

by_gender_age = by_5_year_age.filter(f.col('HeartDisease') == 1).groupBy('Sex', 'age_group').agg(f.count(f.col('HeartDisease')).alias('instances'))
by_gender_age.show() # Make Visualizer

"""
Apart from having less instances of male entries from their 70ss and none in their late 20s, most of their entries from 
their late 30's up until their late 50s, with most female instances at their 60's.
From this, we can assume that heart risk is more prominent with those in their early 40's up until their late 60's.
"""
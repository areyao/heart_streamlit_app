from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)

"""
04_age_and_gender:

    This notebook delves into the age and sex features of the dataset to try and infer insights regarding those with
    heart disease.
"""


"""
We can categorize ages by classifying them within the 5 year age group classification to easily identify 
which groups are more prone to heart failures.
"""

by_5_year_age = get_age_group(heart_data)
five_year_df = by_5_year_age.filter(f.col('HeartDisease') == 1).groupBy('age_group', 'HeartDisease').agg(f.count('HeartDisease').alias('instances')).select("age_group", "instances")
five_year_df.show()

five_year_pd = five_year_df.toPandas().sort_values(by=["instances"],ascending = False).set_index('age_group').plot(kind='barh')
five_year_pd.bar_label(five_year_pd.containers[0])
plt.show()

"""
Through this classification, we can see that those with the most instances of Heart Risk are in their 40s, 50s, and 60s.
We can also see that are only a few instance of heart risk in their early 30's and late 70's.
"""

"""
Apart from the age, we can also look at which gender is more prominent to heart risk.
"""
by_gender = heart_data.filter(f.col('HeartDisease') == 1).groupBy('Sex').agg(f.count(f.col('HeartDisease')).alias('instances')).select("Sex", "instances")
by_gender_pd = by_gender.toPandas().set_index('Sex').plot(kind = "bar")
by_gender_pd.bar_label(by_gender_pd.containers[0])
plt.show()


"""
We can see males have more instances of being at risk than females.
We can look at this alongside the age range to understand how the distributions changes by age.
"""

by_gender_age_pd = by_5_year_age.filter(f.col('HeartDisease') == 1).toPandas().groupby(['age_group', 'Sex']).size()
s_sort = by_gender_age_pd.groupby(level=[0]).sum().sort_values(ascending=False)
ax = by_gender_age_pd.reindex(index=s_sort.index, level=0).unstack().plot(kind = "barh", stacked=True)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.set_xlabel('Number of Entries')
ax.set_ylabel('Age Group')

plt.show()
"""
Apart from having less instances of male entries from their 70s and none in their earlt 20s, most of their entries from 
their late 30's up until their late 50s, with most female instances at their 60's.
From this, we can assume that heart risk is more prominent with those in their early 40's up until their late 60's.
"""
from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
heart_data = get_heart_info(conf)

"""
We can further categorize this by classifying them within the 5 year age group classification to easily identify 
which groups are more prone to heart failures.
"""

by_5_year_age = getAgeGroup(heart_data)
five_year_df = by_5_year_age.filter(f.col('HeartDisease') == 1).groupBy('age_group', 'HeartDisease').agg(f.count('HeartDisease').alias('instances')).select("age_group", "instances")
five_year_df.show()

five_year_pd = five_year_df.toPandas().sort_values(by=["instances"],ascending = False).set_index('age_group').plot(kind='barh')
five_year_pd.bar_label(five_year_pd.containers[0])
plt.show()
# five_year_pd.pivot(index = None, columns="age_group", values="instances").plot(kind='bar', title = "Instances per Age Group")
# plt.show()
"""
Through this classification, we can see that those with the most instances of Heart Risk are in their 40s, 50s, and 60s.
We can also see that is only one instance of heart risk in their early 20's and late 70's.
"""

"""
Apart from the age, we can also look at which gender is more prominent to heart risk.
"""
by_gender = heart_data.filter(f.col('HeartDisease') == 1).groupBy('Sex').agg(f.count(f.col('HeartDisease')).alias('instances')).select("Sex", "instances")
by_gender.show() # Make Visualizer
by_gender_pd = by_gender.toPandas().set_index('Sex').plot(kind = "bar")
by_gender_pd.bar_label(by_gender_pd.containers[0])
plt.show()


"""
We can see males have more instances of being at risk than females.
We can look at this alongside the age range to understand how the distributions changes by age.
"""

# by_gender_age = by_5_year_age.filter(f.col('HeartDisease') == 1).groupBy('Sex', 'age_group').agg(f.count(f.col('HeartDisease')).alias('instances'))
# by_gender_age.show() # Make Visualizer

by_gender_age_pd = by_5_year_age.filter(f.col('HeartDisease') == 1).toPandas().groupby(['age_group', 'Sex']).size()
s_sort = by_gender_age_pd.groupby(level=[0]).sum().sort_values(ascending=False)
ax = by_gender_age_pd.reindex(index=s_sort.index, level=0).unstack().plot(kind = "barh", stacked=True)
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
    # .unstack()
# print(by_gender_age_pd)
# ax = by_gender_age_pd.plot(kind='barh', stacked=True, figsize=(10, 6))
ax.set_xlabel('Number of Entries')
ax.set_ylabel('Age Group')

plt.show()
"""
Apart from having less instances of male entries from their 70ss and none in their late 20s, most of their entries from 
their late 30's up until their late 50s, with most female instances at their 60's.
From this, we can assume that heart risk is more prominent with those in their early 40's up until their late 60's.
"""
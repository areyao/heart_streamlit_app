# from src.model._run_scripts import *
# from src.utils._run_scripts import *
#
# conf = get_conf()
# heart_data = get_heart_info(conf)
# to_pie = conf['required_columns']['categorical_columns']
#
#
# def multi_pie_list(dataframe, cat_cols):
#     list_of_plots = []
#     for colm in range(len(cat_cols)):
#         feat_to_pie = dataframe.groupBy(cat_cols[colm]).count()
#         feat_to_pie = feat_to_pie.toPandas()
#         plt.figure(colm)
#         plt.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[colm]], autopct='%1.1f%%')
#         # feat_to_pie.set_index(cat_cols[colm], inplace=True)
#         # feat_to_pie.plot.pie(y = "count", autopct='%1.0f%%')
#         list_of_plots.append(plt)
#     return list_of_plots
#
#
# plot_list = multi_pie_list(heart_data, to_pie)
# plt.show()
#
# def multi_pie(dataframe, cat_cols):
#     figure, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(2, 3)
#     feat_to_pie = dataframe.groupBy(cat_cols[colm]).count()
#     feat_to_pie = feat_to_pie.toPandas()
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[1]], autopct='%1.0f%%')
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[2]], autopct='%1.0f%%')
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[3]], autopct='%1.0f%%')
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[4]], autopct='%1.0f%%')
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[5]], autopct='%1.0f%%')
#     ax1.pie(feat_to_pie["count"], labels=feat_to_pie[cat_cols[6]], autopct='%1.0f%%')
#     return plt
#
#
# # def multi_pie_plot(dataframe, plot_names, subplot_size=[2, 3]):
# #     figure_size = subplot_size[0] * subplot_size[1]
# #     if len(plot_names) <= figure_size:
# #         figure, axis = plt.subplots(subplot_size[0], subplot_size[1])
# #         for i in range(subplot_size[0]):
# #             for j in range(subplot_size[1]):
# #                 pieplot = (i * 2) + (j + i)
# #                 if pieplot < len(plot_names):
# #                     feat_to_pie = dataframe.groupBy(f.col(plot_names[pieplot])).count()
# #                     feat_to_pie = feat_to_pie.toPandas()
# #                     feat_to_pie.set_index(plot_names[pieplot], inplace=True)
# #                     # list_of_plots.append(feat_to_pie.plot.pie(y="count", autopct='%1.0f%%'))
# #                     # axis[i, j] =
# #                     axis[i,j].set(feat_to_pie.plot.pie(y="count", autopct='%1.0f%%'))
# #                     axis[i, j].set_title(plot_names[pieplot])
# #                 else:
# #                     break
# #     return plt
#
#
# # multi_pie_plot(heart_data, to_pie).show()
# # sex_unique = heart_data.groupBy('Sex').count()
# # sex_unique = sex_unique.toPandas()
# # sex_unique.set_index("Sex",inplace=True)
# # # sex_unique.show()
# # to_pie = conf['required_columns']['categorical_columns']
# # print(sex_unique)
# # sex_plot = sex_unique.plot.pie(y = "count", autopct='%1.0f%%')
#
#
# # .plot(kind='pie', y='count', autopct='%1.0f%%')
# # plt.show()
# # chest_paintype
# # fasting_bs
# # resting_ecg
# # exercise_angina
# # St_slope
# # Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease

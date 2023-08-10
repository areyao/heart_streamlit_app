from src.utils._run_scripts import *
from src.model._run_scripts import *

st.set_page_config(
    page_title = "Model Insights",
    page_icon = ":broken_heart:",
)

st.title('Model Insights')
st.sidebar.success("Model Insights")

shap.initjs()

preprocessor = PipelineModel.load('./pipeline/preprocessor')
base_model = RandomForestClassificationModel.load('./pipeline/base_model')
meta_preprocessor = PipelineModel.load('./pipeline/meta_preprocessor')
meta_model = LogisticRegressionModel.load('./pipeline/meta_model')

conf = get_conf()
heart_data = get_heart_info(conf, handleOutliers= True, streamlitPath = True)
heart_data = preprocessor.transform(heart_data)

# Obtain all features fitted in the vector (categorical vals. fitted with the StringIndexer and continuous values)
x_columns = conf['required_columns']['transformed_columns']

# Obtain their respective feature importance scores
feature_imp = base_model.featureImportances
feat_scores = []
for i in range(len(x_columns)):
    feat_scores.append(float(feature_imp[i]))

feature_importance_df = spark.createDataFrame(zip(x_columns,feat_scores),["Feature","Score"])
feature_importance_df = feature_importance_df.orderBy("Score", ascending = False).toPandas()

fig = px.bar(feature_importance_df, x="Score", y="Feature", orientation='h')
st.plotly_chart(fig, use_container_width=True)

# SHAP Values

# Do the same to create a dataframe from the vectorized values only
x_features = heart_data.select(vector_to_array(f.col("features"))).collect()[0]
x_df = pd.DataFrame(x_features,columns=x_columns)

explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(x_df)

shap_summary = shap.summary_plot(shap_values, features=x_df, feature_names=x_columns)
shap_sum_fig = plt.gcf()

shap_force = shap.force_plot(explainer.expected_value[0], shap_values[0], features=x_df, feature_names=x_columns, matplotlib=True,  figsize=(12,3))
shap_force_fig = plt.gcf()

st.pyplot(shap_sum_fig)
roc_image = Image.open('images/ROC.png')
st.image('images/ROC.png')
# st.pyplot(shap_force_fig)



# modelcoefficients = np.array(meta_model.coefficients)
#
# names=[x["name"] for x in sorted(logistictrainingdata.schema["features"].metadata["ml_attr"]["attrs"]["binary"]+
#    logistictrainingdata.schema["features"].metadata["ml_attr"]["attrs"]["numeric"],
#    key=lambda x: x["idx"])]
#
#
# matchcoefs=np.column_stack((modelcoefficients,np.array(names)))
#
# import pandas as pd
#
# matchcoefsdf=pd.DataFrame(matchcoefs)
#
# matchcoefsdf.columns=['Coefvalue', 'Feature']
#
# print(matchcoefsdf)
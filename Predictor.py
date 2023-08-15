from src.utils._run_scripts import *

# Models for Prediction
preprocessor = PipelineModel.load('./pipeline/preprocessor')
base_model = RandomForestClassificationModel.load('./pipeline/base_model')
meta_preprocessor = PipelineModel.load('./pipeline/meta_preprocessor')
meta_model = LogisticRegressionModel.load('./pipeline/meta_model')

st.set_page_config(
    page_title = "Heart Attack Predictor",
    page_icon = ":broken_heart:",
)

st.title('Heart Attack :broken_heart:')
st.sidebar.success("Heart Attack Predictor")

st.caption("The following form allows a ML model to automatically determine if you have a risk of Heart Disease. "
           "Results may be inaccurate, thus, consulting a medical professional is still highly recommended.")

with st.form(key = 'heart_input'):
    # User Age
    today = dt.datetime.today()
    user_birthDate = st.date_input("Birthdate : ", today, min_value= dt.date(today.year - 100, 1, 1), max_value= today)
    user_age = int(calculate_age(user_birthDate))

    # User sex
    sex_assignments = {'Male': 'M', 'Female': 'F'}
    sex_selection = st.selectbox('Select your sex : ', sex_assignments.keys())
    user_sex = sex_assignments[sex_selection]

    # Diabetes- FastingBS > 120
    diabetes_assignments = {'Yes': 1, 'No': 0}
    diabetes_selection = st.selectbox('Do you have Diabetes? : ', diabetes_assignments.keys())
    user_diabetes = int(diabetes_assignments[diabetes_selection])

    # Exercise Induced Angina - ExrIndAng
    exrindang_assignments = {'Yes': 'Y', 'No': 'N'}
    exrindang_selection = st.selectbox('Does your chest pain when you exercise? : ', exrindang_assignments.keys())
    user_ExrIndAng = exrindang_assignments[exrindang_selection]

    # Medically Consulted Portion

    st.write("The following portion must have medically tested information available.")

    # User BloodPressure - Latest systolic blood pressure (The first number) 90-200
    user_bp = int(st.number_input("Latest Systolic Blood Pressure : ", min_value = 80, max_value = 370, format = '%d'))

    # MaxHeartRate
    user_maxhr = int(st.number_input("Maximum Heart Rate : ", min_value=60, max_value=202, format='%d'))

    #Cholesterol
    user_chol = int(st.number_input("Total Cholesterol : ", min_value = 80, max_value = 370, format = '%d'))

    # Consulted Chestpain Type
    chestpain_assignments = {'Typical Angina': 'TA', 'Atypical Angina': 'ATA', 'Non-anginal pain': 'NAP', 'Asymptomatic': 'ASY'}
    chestpain_selection = st.selectbox('Chest Pain : ', chestpain_assignments.keys())
    user_chestpain = chestpain_assignments[chestpain_selection]

    #RestingElectro
    ecg_assignments = {'Normal': 'Normal', 'Having ST-T wave abnormality': 'ST', "Showing probable or definite left ventricular hypertrophy by Estes' criteria" : 'LVH'}
    ecg_selection = st.selectbox('Resting Electrocardiographic Results: ', ecg_assignments.keys())
    user_ecg = ecg_assignments[ecg_selection]

    # Slope
    slope_assignments = {'Upsloping' : 'Up', 'Flat': 'Flat', 'Downsloping': 'Down'}
    slope_selection = st.selectbox('The slope of the peak exercise ST segment : ', slope_assignments.keys())
    user_slope = slope_assignments[slope_selection]

    # Previous Peak
    user_peak = float(st.slider('Previous Peak', min_value=0.0, max_value=10.0))

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = None

    submit_btn = st.form_submit_button('Run Prediction')
    if submit_btn:
        initial_row = [[user_age, user_sex, user_chestpain, user_bp, user_chol, user_diabetes, user_ecg, user_maxhr,
                       user_ExrIndAng, user_peak, user_slope]]
        # row = spark.sparkContext.parallelize([initial_row])
        st.session_state["user_input"] = initial_row
        predict(preprocessor, base_model, meta_preprocessor, meta_model, initial_row)

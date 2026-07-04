import streamlit as st
import pandas as pd
import joblib

# ---- Load the saved pipeline ----
pipe = joblib.load('titanic_pipeline.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict whether they would have survived.")

# ---- User Inputs ----
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=80, value=25)

with col2:
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"],
                             format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)

# ---- Build input row (must match training column names exactly) ----
family_size = sibsp + parch + 1

input_df = pd.DataFrame([{
    'pclass': pclass,
    'sex': sex,
    'age': age,
    'fare': fare,
    'embarked': embarked,
    'family_size': family_size
}])

st.write("---")

# ---- Predict ----
if st.button("Predict Survival"):
    prediction = pipe.predict(input_df)[0]
    probability = pipe.predict_proba(input_df)[0][1]  # probability of survival (class 1)

    if prediction == 1:
        st.success(f"✅ Likely to SURVIVE — Predicted survival probability: {probability:.1%}")
    else:
        st.error(f"❌ Likely to NOT SURVIVE — Predicted survival probability: {probability:.1%}")

    st.write("### Input Summary")
    st.dataframe(input_df)

st.write("---")
st.caption("Model: Logistic Regression | Trained on the Seaborn Titanic dataset | ~80% test accuracy")

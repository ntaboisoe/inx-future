import streamlit as st


st.set_page_config(page_title="Employee Performance Prediction App", layout="wide")

st.title("INX Future Inc. - Project Summary")

st.markdown("### Employee Performance Prediction Project")
st.markdown("This app is designed to predict employee performance ratings based on various features. "
            "It uses a Random Forest model trained on employee data. Below are the main sections of this application:")
st.markdown("""* [**Main Page:**](https://inx-future-app-ntaboisoe.streamlit.app/main) Provides an overview of the project, source data analysis, and visualizations.""")
st.markdown("""* [**Prediction Model Page:**](https://inx-future-app-ntaboisoe.streamlit.app/prediction_model) Allows users to upload their own data to predict employee performance ratings.""")

st.markdown("**Please Select the Page to Browsse from the Sidebar**")

st.markdown("---")
st.markdown("This project leverages data science techniques for HR analytics, aiming to provide insights into employee performance factors and facilitate data-driven decision-making.")




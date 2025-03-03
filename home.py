import streamlit as st
import gzip
import pickle

# --- Model Loading and Helper Functions (DUPLICATED in each page file) ---
def load_model_package():
    with gzip.open('model_package.pkl.gz', 'rb') as file:
        loaded_model_package = pickle.load(file)
    return loaded_model_package

model_package = load_model_package()
best_model = model_package['model']
encoding_dictionary = model_package['encoding_dictionary']
target_encoded_columns = model_package['target_encoded_columns']
best_features = model_package['best_features']


# --- Helper function for target encoding new data ---
def encode_new_data(df_new, encoding_dictionary_target, target_encoded_columns):
    df_encoded = df_new.copy()
    for col in target_encoded_columns:
        if col in df_encoded.columns: # Check if the column exists in the dataframe
            mapping = encoding_dictionary_target[col]['mapping']
            encoded_values = df_encoded[col].map(mapping)
            # Handle cases where categories in new data are not in the mapping (optional: fill with a default value, or handle as NaN)
            df_encoded[col] = encoded_values.fillna(mapping.mean()) # Using mean for unknown categories as example. Adjust as needed.
        else:
            st.error(f"Required column '{col}' for encoding is missing in uploaded data.")
            return None # Indicate error
    return df_encoded

# --- Helper function to get description from Likert scale value ---
def get_likert_description(value, column_name, encoding_dictionary_likert):
    if column_name in encoding_dictionary_likert and value in encoding_dictionary_likert[column_name]:
        return f"{value} - {encoding_dictionary_likert[column_name][value]}"
    return str(value) # Return value as string if no description found
# --- End of Duplicated Code ---


st.set_page_config(page_title="Employee Performance Prediction App", layout="wide")

st.title("Project Summary")
st.markdown("### Employee Performance Prediction Project")
st.markdown("This app is designed to predict employee performance ratings based on various features. "
            "It uses a Random Forest model trained on employee data. Below are the main sections of this application:")
st.markdown("* **Main Page:** Provides an overview of the project, source data analysis, and visualizations.")
st.markdown("* **Prediction Model Page:** Allows users to upload their own data to predict employee performance ratings.")
st.markdown("---")
st.markdown("This project leverages data science techniques for HR analytics, aiming to provide insights into employee performance factors and facilitate data-driven decision-making.")

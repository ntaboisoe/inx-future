import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


st.title("Main Page")
st.markdown("## Project Main Overview")
st.markdown("Explore the different facets of our Employee Performance Prediction project using the sub-pages in the sidebar under 'Main Page Activities'.")

main_page_activity = st.selectbox("Main Page Activities", ["Overview", "Source Data Analysis", "Visualizations"])

if main_page_activity == "Source Data Analysis":
    st.header("Source Data Analysis")

    # --- Source Data Analysis Sub-page Content ---
    # ... (rest of Source Data Analysis content - unchanged) ...


elif main_page_activity == "Visualizations":
    st.header("Visualizations")

    # --- Visualizations Sub-page Content ---
    # ... (rest of Visualizations content - unchanged) ...

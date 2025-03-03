import streamlit as st
import pandas as pd
import io  # For handling in-memory file download
from home.home import best_model, encoding_dictionary, target_encoded_columns, best_features, encode_new_data, get_likert_description # Import from home.py

st.title("Prediction Model")
st.markdown("## Employee Performance Prediction")
st.markdown("Upload an Excel file containing employee data to predict performance ratings.")

uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_excel(uploaded_file)
        st.success("File uploaded and processed successfully!")

        # --- Feature Column Check ---
        missing_features = [feature for feature in best_features if feature not in df_uploaded.columns]
        if missing_features:
            st.error(f"Error: The uploaded file is missing the following required feature columns: {', '.join(missing_features)}")
        else:
            st.info("Required feature columns are present. Proceeding with prediction.")

            # --- Extract Features ---
            X_new = df_uploaded[best_features].copy() # Use .copy() to avoid SettingWithCopyWarning

            # --- Encode Categorical Features ---
            X_encoded_new = encode_new_data(X_new, encoding_dictionary, target_encoded_columns)

            if X_encoded_new is not None: # Proceed only if encoding was successful
                # --- Make Predictions ---
                predictions = best_model.predict(X_encoded_new)
                df_predictions = df_uploaded.copy() # Copy original uploaded data
                df_predictions['PredictedPerformanceRating'] = predictions

                # --- Add Likert Scale Descriptions ---
                likert_scale_features = ['EmpEnvironmentSatisfaction', 'EmpWorkLifeBalance'] # Define likert_scale_features here as it is used only in this page now. Or import if needed in multiple pages.
                target_column = 'PerformanceRating' # Define target_column here or import if needed in multiple pages
                likert_cols_for_desc = likert_scale_features + [target_column, 'PredictedPerformanceRating']
                for col in likert_cols_for_desc:
                    if col in df_predictions.columns: # Check if column exists
                        df_predictions[f'{col} Description'] = df_predictions[col].apply(lambda x: get_likert_description(x, col, encoding_dictionary))

                # --- Display Predictions DataFrame ---
                st.subheader("Predictions")
                st.dataframe(df_predictions)

                # --- Download Button ---
                csv_buffer = io.StringIO()
                df_predictions.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name="employee_performance_predictions.csv",
                    mime='text/csv'
                )

                # --- Re-upload/Exit Buttons ---
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Re-upload File"):
                        st.session_state.clear() # Clear session state to re-run from file upload. This might be too aggressive if you are using session state for other things. A more targeted approach might be to reset just the file uploader. For simple case this clear all works.
                        st.rerun() # Force rerun to clear file uploader and start fresh.
                with col2:
                    if st.button("Exit Application"):
                        st.stop() # Stop Streamlit app execution.


    except Exception as e:
        st.error(f"Error processing file: {e}")

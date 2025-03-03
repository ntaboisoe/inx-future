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

    st.subheader("Import Libraries")
    st.code("""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
""", language='python')

    st.subheader("Data Import")
    st.code("""
iris_df = sns.load_dataset('iris')
st.dataframe(iris_df.head())
""", language='python')
    iris_df = sns.load_dataset('iris')
    st.dataframe(iris_df.head())

    st.subheader("Exploratory Data Analysis")
    st.markdown("Descriptive Statistics:")
    st.dataframe(iris_df.describe())

    st.markdown("Pair Plot:")
    fig = sns.pairplot(iris_df, hue='species')
    st.pyplot(fig)
    plt.clf() # Clear pyplot for next plot

    st.subheader("Data Preprocessing")
    st.markdown("Encoding Categorical Variables (Example using Label Encoding for 'species'):")
    st.code("""
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
iris_df['species_encoded'] = label_encoder.fit_transform(iris_df['species'])
st.dataframe(iris_df[['species', 'species_encoded']].head())
""", language='python')
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    iris_df['species_encoded'] = label_encoder.fit_transform(iris_df['species'])
    st.dataframe(iris_df[['species', 'species_encoded']].head())


    st.subheader("Model Building (Example with Logistic Regression)")
    st.markdown("Model Training:")
    st.code("""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
""", language='python')
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['species_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    st.markdown("Model Evaluation:")
    st.code("""
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
""", language='python')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)


    st.subheader("Conclusion")
    st.markdown("This section provided a basic overview of a data science workflow using the Iris dataset. You can expand on each step to perform more in-depth analysis and modeling.")


elif main_page_activity == "Visualizations":
    st.header("Visualizations")

    # --- Visualizations Sub-page Content ---

    st.subheader("Attrition Analysis (Example - Species Distribution)")
    st.markdown("Bar Plot of Species:")
    fig_species_count = sns.countplot(x='species', data=iris_df)
    st.pyplot(fig_species_count.figure)
    plt.clf()

    st.subheader("Performance Analysis (Example - Feature Distributions by Species)")
    for feature in iris_df.columns[:-1]: # Exclude 'species' and 'species_encoded'
        fig_dist = sns.boxplot(x='species', y=feature, data=iris_df)
        st.pyplot(fig_dist.figure)
        plt.clf()

    st.subheader("Satisfaction Analysis (Example - Correlation Heatmap)")
    st.markdown("Correlation Heatmap:")
    corr_matrix = iris_df.corr(numeric_only=True)
    fig_heatmap = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(fig_heatmap)
    plt.clf()

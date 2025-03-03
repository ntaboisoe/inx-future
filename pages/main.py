import streamlit as st
from ..home import model_package # Just import model_package for now

st.title("Testing Import")
print("Import Attempted") # Add this print statement

if 'model_package' in locals(): # Check if import was successful
    st.success("Successfully imported model_package from home.py!")
else:
    st.error("Failed to import model_package. Check terminal for errors.")




import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ..home import model_package, best_model, encoding_dictionary, target_encoded_columns, best_features, encode_new_data, get_likert_description # Corrected import



# --- Unpack model package (if needed here, example visualizations may not directly need the model itself) ---
# best_model = model_package['model'] # Not used in this example, but keep if you plan to add model related stuff here
# encoding_dictionary = model_package['encoding_dictionary']
# target_encoded_columns = model_package['target_encoded_columns']
# best_features = model_package['best_features']


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

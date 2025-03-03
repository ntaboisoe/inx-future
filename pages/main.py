#-------------------------------------------------------------------------------------------------------------------------
# Step (00) - Import the Libraries to be Used in the Streamlit App
#-------------------------------------------------------------------------------------------------------------------------

# General Libraries
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Statistical data visualization

# Libraries for Reading Data from Website
import requests  # Sending HTTP requests
from io import BytesIO  # Handling byte streams

# Libraries for Encoding Categorical Variables
from category_encoders import TargetEncoder  # Target mean encoding of categorical features

# Libraries for Machine Learning Models
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier

# Libraries for Model Evaluation
from sklearn.metrics import accuracy_score  # Accuracy metri
from sklearn.metrics import classification_report  # Classification report
from sklearn.metrics import confusion_matrix  # Confusion matrix

# Libraries for Model Selection and Hyperparameter Tuning
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.model_selection import RandomizedSearchCV  # Hyperparameter tuning using randomized search

#-------------------------------------------------------------------------------------------------------------------------
# Step (01) - Define the Page Headers
#-------------------------------------------------------------------------------------------------------------------------

st.title("INX Future Inc. Master Data Exploration")
st.markdown("## Project Main Overview")

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Import INX Data
#-------------------------------------------------------------------------------------------------------------------------

st.markdown("Before we begin let us load the INX Master Data Set that was used to train the model.")

@st.cache_data #To prevent multiple loading of the dataframe
def load_data():
    successful_import = 0
    #Import the Employee Data
    url = 'https://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
    response = requests.get(url)
    
    if response.status_code == 200:
        file = BytesIO(response.content)
        df = pd.read_excel(io = file, sheet_name = 'INX_Future_Inc_Employee_Perform')
        st.write('Data imported successfully')
        successful_import = 1
        return df,successful_import
    else:
        st.write(f'Error: {response.status_code}')
        successful_import = 0
        return None,successful_import
    

df,successful_import = load_data()

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Allow User to now Select Subpages
#-------------------------------------------------------------------------------------------------------------------------
if successful_import == 0:
    st.write('Data import was not successful. Please reload the app and try again')
    
elif successful_import == 1:
    st.write('Explore the different facets of our Employee Performance Prediction project using the sub-pages in the dropdown below')
    
    main_page_activity = st.selectbox("Select Page to Explore", ["Source Data Analysis", "Source Data Visualizations"])

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Load Source Data Analysis Subpage Based on User Selection
#-------------------------------------------------------------------------------------------------------------------------

if main_page_activity == "Source Data Analysis":
    st.header("Source Data Analysis")

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Import Libraries Section of 1st Page as Narkdown
#-------------------------------------------------------------------------------------------------------------------------    

    libraries_import_markdown ="""
        # General Libraries
        import pandas as pd  # Data manipulation and analysis
        import matplotlib.pyplot as plt  # Data visualization
        import seaborn as sns  # Statistical data visualization
        
        # Libraries for Reading Data from Website
        import requests  # Sending HTTP requests
        from io import BytesIO  # Handling byte streams
        
        # Libraries for Encoding Categorical Variables
        from category_encoders import TargetEncoder  # Target mean encoding of categorical features
        
        # Libraries for Machine Learning Models
        from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
        
        # Libraries for Model Evaluation
        from sklearn.metrics import accuracy_score  # Accuracy metri
        from sklearn.metrics import classification_report  # Classification report
        from sklearn.metrics import confusion_matrix  # Confusion matrix
        
        # Libraries for Model Selection and Hyperparameter Tuning
        from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
        from sklearn.model_selection import RandomizedSearchCV  # Hyperparameter tuning using randomized search
    """

    st.subheader("Import Libraries")
    st.code(libraries_import_markdown)
    st.code("""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
""", language='python')

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Import Data Section of 1st Page as Markdown
#-------------------------------------------------------------------------------------------------------------------------    

    st.subheader("Data Import")
    data_import_markdown = """
        def load_data():
            successful_import = 0
            #Import the Employee Data
            url = 'https://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
            response = requests.get(url)
            
            if response.status_code == 200:
                file = BytesIO(response.content)
                df = pd.read_excel(io = file, sheet_name = 'INX_Future_Inc_Employee_Perform')
                st.write('Data imported successfully')
                successful_import = 1
                return df,successful_import
            else:
                st.write(f'Error: {response.status_code}')
                successful_import = 0
                return None,successful_import
            
        
        df,successful_import = load_data()    
    """
    st.code(data_import_markdown)
    st.code("""
iris_df = sns.load_dataset('iris')
st.dataframe(iris_df.head())
""", language='python')

    st.dataframe(df.head().T)
    
    iris_df = sns.load_dataset('iris')
    st.dataframe(iris_df.head())
#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - EDA for Imported Data
#-------------------------------------------------------------------------------------------------------------------------    
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


elif main_page_activity == "Source Data Visualizations":
    st.header("Source Data Visualizations")

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



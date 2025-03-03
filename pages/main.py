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

#Library for the app
import streamlit as st

#-------------------------------------------------------------------------------------------------------------------------
# Step (01) - Define the Page Headers
#-------------------------------------------------------------------------------------------------------------------------

st.title("INX Future Inc. Master Data Exploration")
st.markdown("## Project Main Overview")

#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Import INX Data
#-------------------------------------------------------------------------------------------------------------------------

st.markdown("Before we begin let us load the INX Master Data Set that was used to train the model.")


@st.cache_data  # To prevent multiple loading of the dataframe
def load_data():
    successful_import = 0
    # Import the Employee Data
    url = 'https://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
    response = requests.get(url)
    
    if response.status_code == 200:
        file = BytesIO(response.content)
        df = pd.read_excel(io=file)
        st.write('Data imported successfully')  # Success message
        successful_import = 1
        return df, successful_import
    else:
        st.write(f'Error: {response.status_code}')  # Error message
        successful_import = 0
        return None, successful_import

# Call the load_data function to import the data
df, successful_import = load_data()

# Step (02) - Conditional Display of Selection Box after Successful Data Import






#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Allow User to now Select Subpages
#-------------------------------------------------------------------------------------------------------------------------
if successful_import == 0:
    st.write('Data import was not successful. Please reload the app and try again.')
    
elif successful_import == 1:
    st.write('Explore the different facets of our Employee Performance Prediction project using the sub-pages in the dropdown below.')
    
    # Display the selection box to allow user to select subpages, only if import is successful
    main_page_activity = st.selectbox("Select Page to Explore", ["Select Page", "Source Data Analysis", "Source Data Visualizations"])
    
#-------------------------------------------------------------------------------------------------------------------------
# Step (02) - Load Source Data Analysis Subpage Based on User Selection
#-------------------------------------------------------------------------------------------------------------------------
    # Additional logic to render content based on the selected page (you can expand this part as needed)
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
    
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Import Data Section of 1st Page as Markdown
    #-------------------------------------------------------------------------------------------------------------------------    
    
        st.subheader("Data Import")
        data_import_markdown = """
            def load_data():
                successful_import = 0
                # Import the Employee Data
                url = 'https://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls'
                response = requests.get(url)
                
                if response.status_code == 200:
                    file = BytesIO(response.content)
                    df = pd.read_excel(io=file)
                    st.write('Data imported successfully')  # Success message
                    successful_import = 1
                    return df, successful_import
                else:
                    st.write(f'Error: {response.status_code}')  # Error message
                    successful_import = 0
                    return None, successful_import
            
            # Call the load_data function to import the data
            df, successful_import = load_data()        
        """
        st.code(data_import_markdown)
        st.dataframe(df.head().T)
        
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - EDA for Imported Data
    #-------------------------------------------------------------------------------------------------------------------------    
        st.subheader("Exploratory Data Analysis")
        st.markdown("Descriptive Statistics:")
        st.dataframe(df.describe())

    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Data Preprocessing - Select Columns
    #-------------------------------------------------------------------------------------------------------------------------            
    
        st.subheader("Data Preprocessing")
        st.write("Extract Feature and Target Columns")

        target_encoded_columns = ['EmpJobRole', 'EmpDepartment']
        already_encoded_columns = ['EmpWorkLifeBalance', 'EmpEnvironmentSatisfaction']
        likert_scale_features = ['EmpEnvironmentSatisfaction', 'EmpWorkLifeBalance']
        best_features = [
            'EmpEnvironmentSatisfaction',
            'EmpLastSalaryHikePercent',
            'EmpJobRole',
            'YearsSinceLastPromotion',
            'ExperienceYearsInCurrentRole',
            'EmpDepartment',
            'EmpWorkLifeBalance'
        ]
        target_column = 'PerformanceRating'
        
        preprocessing_code_markdown="""
                target_encoded_columns = ['EmpJobRole', 'EmpDepartment']
                already_encoded_columns = ['EmpWorkLifeBalance', 'EmpEnvironmentSatisfaction']
                likert_scale_features = ['EmpEnvironmentSatisfaction', 'EmpWorkLifeBalance']
                best_features = [
                    'EmpEnvironmentSatisfaction',
                    'EmpLastSalaryHikePercent',
                    'EmpJobRole',
                    'YearsSinceLastPromotion',
                    'ExperienceYearsInCurrentRole',
                    'EmpDepartment',
                    'EmpWorkLifeBalance'
                ]
        target_column = 'PerformanceRating'
                """
        st.code(preprocessing_code_markdown)
        
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Data Preprocessing - Create DF with these columns
    #-------------------------------------------------------------------------------------------------------------------------                 
        X = df[best_features]
        y = df[target_column]

        create_variables_markdown="""
            X = df[best_features]
            y = df[target_column]
            """
        st.write("Create Target and Features Variables")
        st.code(create_variables_markdown)
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Encode the Target Variables
    #-------------------------------------------------------------------------------------------------------------------------     
        encoder = TargetEncoder(cols=target_encoded_columns)
        X_encoded = encoder.fit_transform(X, y) # Fit and transform in one step

        encode_data_markdown="""
            encoder = TargetEncoder(cols=target_encoded_columns)
            X_encoded = encoder.fit_transform(X, y) # Fit and transform in one step
        """
        st.write("Encode the Data")
        st.code(encode_data_markdown)
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Train the Model
    #-------------------------------------------------------------------------------------------------------------------------   
        st.subheader("Model Building Random Forest")
        st.markdown("Model Training:")

        # Split data into training and testing sets (optional but good practice for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        best_model = RandomForestClassifier(random_state=42) # Initialize the model
        best_model.fit(X_train, y_train) # Train the model
        
        training_markdown="""
            # Split data into training and testing sets (optional but good practice for evaluation)
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            best_model = RandomForestClassifier(random_state=42) # Initialize the model
            best_model.fit(X_train, y_train) # Train the model
            """
        st.code(training_markdown)
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Display Accuracy Score and Confusion Matrix
    #-------------------------------------------------------------------------------------------------------------------------
        st.markdown("Model Evaluation:")
        y_pred = best_model.predict(X_test) # Make predictions on the test set
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        st.write(f"Accuracy of the Best Model: {accuracy:.4f}")
        st.write("\nConfusion Matrix:")
        st.write(conf_matrix)
    #-------------------------------------------------------------------------------------------------------------------------
    # Step (02) - Conclusion
    #-------------------------------------------------------------------------------------------------------------------------
    
        st.subheader("Conclusion")
        st.markdown("This section provided a basic overview of a data science workflow using the INX Employee Data.")
    
        
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
    
    

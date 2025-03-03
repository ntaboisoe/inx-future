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
        
        emp_education_level_map = {
            1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'
        }
        emp_environment_satisfaction_map = {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        }
        emp_job_involvement_map = {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        }
        emp_job_satisfaction_map = {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        }
        performance_rating_map = {
            1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'
        }
        emp_relationship_satisfaction_map = {
            1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'
        }
        emp_work_life_balance_map = {
            1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'
        }
        
        
        # Function to map encoded categorical values to descriptions for plotting
        def map_encoded_values(series, mapping):
            return series.map(mapping)
        
        # Apply mappings to the dataframe for display purposes in visualizations
        df['EmpEducationLevel_Descr'] = map_encoded_values(df['EmpEducationLevel'], emp_education_level_map)
        df['EmpEnvironmentSatisfaction_Descr'] = map_encoded_values(df['EmpEnvironmentSatisfaction'], emp_environment_satisfaction_map)
        df['EmpJobInvolvement_Descr'] = map_encoded_values(df['EmpJobInvolvement'], emp_job_involvement_map)
        df['EmpJobSatisfaction_Descr'] = map_encoded_values(df['EmpJobSatisfaction'], emp_job_satisfaction_map)
        df['PerformanceRating_Descr'] = map_encoded_values(df['PerformanceRating'], performance_rating_map)
        df['EmpRelationshipSatisfaction_Descr'] = map_encoded_values(df['EmpRelationshipSatisfaction'], emp_relationship_satisfaction_map)
        df['EmpWorkLifeBalance_Descr'] = map_encoded_values(df['EmpWorkLifeBalance'], emp_work_life_balance_map)
        
        # --- Theme 1: Departmental Performance Analysis ---
        st.header("Departmental Performance Analysis")
        
        # 1. Average Performance Rating by Department
        st.subheader("Average Performance Rating by Department")
        st.markdown("Bar chart showing average performance rating for each department.")
        
        avg_performance_dept = df.groupby('EmpDepartment')['PerformanceRating'].mean().sort_values()
        
        fig_avg_perf_dept, ax_avg_perf_dept = plt.subplots(figsize=(10, 6))
        avg_performance_dept.plot(kind='barh', ax=ax_avg_perf_dept, color='skyblue')
        ax_avg_perf_dept.set_xlabel('Average Performance Rating')
        ax_avg_perf_dept.set_ylabel('Department')
        ax_avg_perf_dept.set_title('Average Performance Rating by Department')
        st.pyplot(fig_avg_perf_dept)
        
        
        # 2. Average Performance Rating by Department, broken down by Attrition
        st.subheader("Average Performance Rating by Department, broken down by Attrition")
        st.markdown("Grouped bar chart showing average performance for each department, separated by attrition status.")
        
        avg_performance_dept_attrition = df.groupby(['EmpDepartment', 'Attrition'])['PerformanceRating'].mean().unstack()
        
        fig_avg_perf_dept_attrition, ax_avg_perf_dept_attrition = plt.subplots(figsize=(12, 7))
        avg_performance_dept_attrition.plot(kind='bar', ax=ax_avg_perf_dept_attrition, color=['lightcoral', 'lightseagreen'])
        ax_avg_perf_dept_attrition.set_xlabel('Department')
        ax_avg_perf_dept_attrition.set_ylabel('Average Performance Rating')
        ax_avg_perf_dept_attrition.set_title('Average Performance Rating by Department and Attrition')
        ax_avg_perf_dept_attrition.legend(title='Attrition')
        st.pyplot(fig_avg_perf_dept_attrition)
        
        
        # 3. Average Salary Hike Percent by Department
        st.subheader("Average Salary Hike Percent by Department")
        st.markdown("Bar chart showing average salary hike percentage for each department.")
        
        avg_salary_hike_dept = df.groupby('EmpDepartment')['EmpLastSalaryHikePercent'].mean().sort_values()
        
        fig_avg_salary_hike_dept, ax_avg_salary_hike_dept = plt.subplots(figsize=(10, 6))
        avg_salary_hike_dept.plot(kind='barh', ax=ax_avg_salary_hike_dept, color='lightgreen')
        ax_avg_salary_hike_dept.set_xlabel('Average Salary Hike Percent')
        ax_avg_salary_hike_dept.set_ylabel('Department')
        ax_avg_salary_hike_dept.set_title('Average Salary Hike Percent by Department')
        st.pyplot(fig_avg_salary_hike_dept)
        
        
        # 4. Gender Distribution by Department
        st.subheader("Gender Distribution by Department")
        st.markdown("Stacked bar chart showing the proportion of genders in each department.")
        
        gender_dept_distribution = df.groupby(['EmpDepartment', 'Gender']).size().unstack(fill_value=0)
        gender_dept_proportion = gender_dept_distribution.div(gender_dept_distribution.sum(axis=1), axis=0)
        
        fig_gender_dept, ax_gender_dept = plt.subplots(figsize=(12, 7))
        gender_dept_proportion.plot(kind='bar', stacked=True, ax=ax_gender_dept, color=['lightblue', 'pink'])
        ax_gender_dept.set_xlabel('Department')
        ax_gender_dept.set_ylabel('Proportion of Gender')
        ax_gender_dept.set_title('Gender Distribution by Department')
        ax_gender_dept.legend(title='Gender')
        st.pyplot(fig_gender_dept)
        
        
        # 5. Average Years at Company by Department
        st.subheader("Average Years at Company by Department by Gender")
        st.markdown("Bar chart showing average years at the company for each department, grouped by gender.")
        
        avg_years_company_dept_gender = df.groupby(['EmpDepartment', 'Gender'])['ExperienceYearsAtThisCompany'].mean().unstack()
        
        fig_years_company_dept_gender, ax_years_company_dept_gender = plt.subplots(figsize=(12, 7))
        avg_years_company_dept_gender.plot(kind='bar', ax=ax_years_company_dept_gender, color=['skyblue', 'salmon'])
        ax_years_company_dept_gender.set_xlabel('Department')
        ax_years_company_dept_gender.set_ylabel('Average Years at Company')
        ax_years_company_dept_gender.set_title('Average Years at Company by Department and Gender')
        ax_years_company_dept_gender.legend(title='Gender')
        st.pyplot(fig_years_company_dept_gender)
        
        
        
        # --- Theme 2: Satisfaction and Performance ---
        st.header("Employee Satisfaction and Performance")
        
        # 6. Performance Rating vs. Job Satisfaction
        st.subheader("Performance Rating vs. Job Satisfaction")
        st.markdown("Box plot showing the distribution of performance ratings for each level of job satisfaction.")
        
        fig_perf_job_satisfaction, ax_perf_job_satisfaction = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpJobSatisfaction_Descr', y='PerformanceRating', data=df,
                    order=['Low', 'Medium', 'High', 'Very High'], ax=ax_perf_job_satisfaction, palette="viridis") # Ordering based on encoded value meaning
        ax_perf_job_satisfaction.set_xlabel('Job Satisfaction')
        ax_perf_job_satisfaction.set_ylabel('Performance Rating')
        ax_perf_job_satisfaction.set_title('Performance Rating vs. Job Satisfaction')
        st.pyplot(fig_perf_job_satisfaction)
        
        
        # 7. Performance Rating vs. Environment Satisfaction
        st.subheader("Performance Rating vs. Environment Satisfaction")
        st.markdown("Box plot showing the distribution of performance ratings for each level of environment satisfaction.")
        
        fig_perf_env_satisfaction, ax_perf_env_satisfaction = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpEnvironmentSatisfaction_Descr', y='PerformanceRating', data=df,
                    order=['Low', 'Medium', 'High', 'Very High'], ax=ax_perf_env_satisfaction, palette="plasma") # Ordering based on encoded value meaning
        ax_perf_env_satisfaction.set_xlabel('Environment Satisfaction')
        ax_perf_env_satisfaction.set_ylabel('Performance Rating')
        ax_perf_env_satisfaction.set_title('Performance Rating vs. Environment Satisfaction')
        st.pyplot(fig_perf_env_satisfaction)
        
        
        # 8. Performance Rating vs. Work-Life Balance
        st.subheader("Performance Rating vs. Work-Life Balance")
        st.markdown("Box plot showing the distribution of performance ratings for each level of work-life balance.")
        
        fig_perf_wlb, ax_perf_wlb = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpWorkLifeBalance_Descr', y='PerformanceRating', data=df,
                    order=['Bad', 'Good', 'Better', 'Best'], ax=ax_perf_wlb, palette="magma") # Ordering based on encoded value meaning
        ax_perf_wlb.set_xlabel('Work Life Balance')
        ax_perf_wlb.set_ylabel('Performance Rating')
        ax_perf_wlb.set_title('Performance Rating vs. Work Life Balance')
        st.pyplot(fig_perf_wlb)
        
        
        # 9. Performance Rating vs. Relationship Satisfaction
        st.subheader("Performance Rating vs. Relationship Satisfaction")
        st.markdown("Box plot showing the distribution of performance ratings for each level of relationship satisfaction.")
        
        fig_perf_relationship_satisfaction, ax_perf_relationship_satisfaction = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpRelationshipSatisfaction_Descr', y='PerformanceRating', data=df,
                    order=['Low', 'Medium', 'High', 'Very High'], ax=ax_perf_relationship_satisfaction, palette="cividis") # Ordering based on encoded value meaning
        ax_perf_relationship_satisfaction.set_xlabel('Relationship Satisfaction')
        ax_perf_relationship_satisfaction.set_ylabel('Performance Rating')
        ax_perf_relationship_satisfaction.set_title('Performance Rating vs. Relationship Satisfaction')
        st.pyplot(fig_perf_relationship_satisfaction)
        
        
        
        # --- Theme 3: Experience and Performance ---
        st.header("Experience and Performance")
        
        # 10. Performance Rating vs. Total Work Experience
        st.subheader("Performance Rating vs. Total Work Experience")
        st.markdown("Scatter plot with regression line showing the relationship between total work experience and performance rating.")
        
        fig_perf_total_exp, ax_perf_total_exp = plt.subplots(figsize=(10, 6))
        sns.regplot(x='TotalWorkExperienceInYears', y='PerformanceRating', data=df, ax=ax_perf_total_exp, color='purple')
        ax_perf_total_exp.set_xlabel('Total Work Experience (Years)')
        ax_perf_total_exp.set_ylabel('Performance Rating')
        ax_perf_total_exp.set_title('Performance Rating vs. Total Work Experience')
        st.pyplot(fig_perf_total_exp)
        
        
        # 11. Performance Rating vs. Years at Company
        st.subheader("Performance Rating vs. Years at Company")
        st.markdown("Scatter plot with regression line showing the relationship between years at company and performance rating.")
        
        fig_perf_years_at_company, ax_perf_years_at_company = plt.subplots(figsize=(10, 6))
        sns.regplot(x='ExperienceYearsAtThisCompany', y='PerformanceRating', data=df, ax=ax_perf_years_at_company, color='orange')
        ax_perf_years_at_company.set_xlabel('Years at Company')
        ax_perf_years_at_company.set_ylabel('Performance Rating')
        ax_perf_years_at_company.set_title('Performance Rating vs. Years at Company')
        st.pyplot(fig_perf_years_at_company)
        
        
        # 12. Performance Rating vs. Years in Current Role
        st.subheader("Performance Rating vs. Years in Current Role")
        st.markdown("Scatter plot with regression line showing the relationship between years in current role and performance rating.")
        
        fig_perf_years_in_role, ax_perf_years_in_role = plt.subplots(figsize=(10, 6))
        sns.regplot(x='ExperienceYearsInCurrentRole', y='PerformanceRating', data=df, ax=ax_perf_years_in_role, color='brown')
        ax_perf_years_in_role.set_xlabel('Years in Current Role')
        ax_perf_years_in_role.set_ylabel('Performance Rating')
        ax_perf_years_in_role.set_title('Performance Rating vs. Years in Current Role')
        st.pyplot(fig_perf_years_in_role)
        
        
        # 13. Performance Rating vs. Years with Current Manager
        st.subheader("Performance Rating vs. Years with Current Manager")
        st.markdown("Scatter plot with regression line showing the relationship between years with current manager and performance rating.")
        
        fig_perf_years_w_manager, ax_perf_years_w_manager = plt.subplots(figsize=(10, 6))
        sns.regplot(x='YearsWithCurrManager', y='PerformanceRating', data=df, ax=ax_perf_years_w_manager, color='grey')
        ax_perf_years_w_manager.set_xlabel('Years with Current Manager')
        ax_perf_years_w_manager.set_ylabel('Performance Rating')
        ax_perf_years_w_manager.set_title('Performance Rating vs. Years with Current Manager')
        st.pyplot(fig_perf_years_w_manager)
        
        
        
        # --- Theme 4: Education, Training and Performance ---
        st.header("Education, Training and Performance")
        
        # 14. Performance Rating vs. Education Level
        st.subheader("Performance Rating vs. Education Level")
        st.markdown("Box plot showing the distribution of performance ratings for each education level.")
        
        fig_perf_education_level, ax_perf_education_level = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpEducationLevel_Descr', y='PerformanceRating', data=df,
                    order=['Below College', 'College', 'Bachelor', 'Master', 'Doctor'], ax=ax_perf_education_level, palette="Set2") # Ordering based on encoded value meaning
        ax_perf_education_level.set_xlabel('Education Level')
        ax_perf_education_level.set_ylabel('Performance Rating')
        ax_perf_education_level.set_title('Performance Rating vs. Education Level')
        st.pyplot(fig_perf_education_level)
        
        
        # 15. Performance Rating vs. Training Times Last Year
        st.subheader("Performance Rating vs. Training Times Last Year")
        st.markdown("Box plot showing the distribution of performance ratings for different training times last year.")
        
        fig_perf_training_times, ax_perf_training_times = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='TrainingTimesLastYear', y='PerformanceRating', data=df, ax=ax_perf_training_times, palette="Paired")
        ax_perf_training_times.set_xlabel('Training Times Last Year')
        ax_perf_training_times.set_ylabel('Performance Rating')
        ax_perf_training_times.set_title('Performance Rating vs. Training Times Last Year')
        st.pyplot(fig_perf_training_times)
        
        
        
        # --- Theme 5: Work Environment and Performance ---
        st.header("Work Environment and Performance")
        
        # 16. Performance Rating vs. Business Travel Frequency
        st.subheader("Performance Rating vs. Business Travel Frequency")
        st.markdown("Box plot showing the distribution of performance ratings for each business travel frequency category.")
        
        fig_perf_travel_freq, ax_perf_travel_freq = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='BusinessTravelFrequency', y='PerformanceRating', data=df, ax=ax_perf_travel_freq, palette="Accent")
        ax_perf_travel_freq.set_xlabel('Business Travel Frequency')
        ax_perf_travel_freq.set_ylabel('Performance Rating')
        ax_perf_travel_freq.set_title('Performance Rating vs. Business Travel Frequency')
        st.pyplot(fig_perf_travel_freq)
        
        
        # 17. Performance Rating vs. Distance From Home
        st.subheader("Performance Rating vs. Distance From Home")
        st.markdown("Scatter plot with regression line showing the relationship between distance from home and performance rating.")
        
        fig_perf_distance_home, ax_perf_distance_home = plt.subplots(figsize=(10, 6))
        sns.regplot(x='DistanceFromHome', y='PerformanceRating', data=df, ax=ax_perf_distance_home, color='teal')
        ax_perf_distance_home.set_xlabel('Distance From Home')
        ax_perf_distance_home.set_ylabel('Performance Rating')
        ax_perf_distance_home.set_title('Performance Rating vs. Distance From Home')
        st.pyplot(fig_perf_distance_home)
        
        
        # 18. Performance Rating vs. Overtime
        st.subheader("Performance Rating vs. OverTime")
        st.markdown("Box plot showing the distribution of performance ratings for employees with and without overtime.")
        
        fig_perf_overtime, ax_perf_overtime = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='OverTime', y='PerformanceRating', data=df, ax=ax_perf_overtime, palette="Set3")
        ax_perf_overtime.set_xlabel('OverTime')
        ax_perf_overtime.set_ylabel('Performance Rating')
        ax_perf_overtime.set_title('Performance Rating vs. OverTime')
        st.pyplot(fig_perf_overtime)
        
        
        # 19. Performance Rating vs. Hourly Rate
        st.subheader("Performance Rating vs. Hourly Rate")
        st.markdown("Scatter plot with regression line showing the relationship between hourly rate and performance rating.")
        
        fig_perf_hourly_rate, ax_perf_hourly_rate = plt.subplots(figsize=(10, 6))
        sns.regplot(x='EmpHourlyRate', y='PerformanceRating', data=df, ax=ax_perf_hourly_rate, color='indigo')
        ax_perf_hourly_rate.set_xlabel('Hourly Rate')
        ax_perf_hourly_rate.set_ylabel('Performance Rating')
        ax_perf_hourly_rate.set_title('Performance Rating vs. Hourly Rate')
        st.pyplot(fig_perf_hourly_rate)
        
        
        # --- Theme 6: Job Involvement and Performance ---
        st.header("Job Involvement and Performance")
        
        # 20. Performance Rating vs. Job Involvement
        st.subheader("Performance Rating vs. Job Involvement")
        st.markdown("Box plot showing the distribution of performance ratings for each level of job involvement.")
        
        fig_perf_job_involvement, ax_perf_job_involvement = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='EmpJobInvolvement_Descr', y='PerformanceRating', data=df,
                    order=['Low', 'Medium', 'High', 'Very High'], ax=ax_perf_job_involvement, palette="viridis") # Ordering based on encoded value meaning
        ax_perf_job_involvement.set_xlabel('Job Involvement')
        ax_perf_job_involvement.set_ylabel('Performance Rating')
        ax_perf_job_involvement.set_title('Performance Rating vs. Job Involvement')
        st.pyplot(fig_perf_job_involvement)
        
        
        # --- Theme 7: Tenure and Company Change ---
        st.header("Tenure, Company Change and Performance")
        
        # 21. Performance Rating vs. Number of Companies Worked
        st.subheader("Performance Rating vs. Number of Companies Worked")
        st.markdown("Box plot showing the distribution of performance ratings for different number of companies worked.")
        
        fig_perf_num_companies, ax_perf_num_companies = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='NumCompaniesWorked', y='PerformanceRating', data=df, ax=ax_perf_num_companies, palette="Set1") # Removed '_Descr' and order parameter
        ax_perf_num_companies.set_xlabel('Number of Companies Worked') # Corrected label
        ax_perf_num_companies.set_ylabel('Performance Rating')
        ax_perf_num_companies.set_title('Performance Rating vs. Number of Companies Worked')
        st.pyplot(fig_perf_num_companies)
        
        
        # --- Theme 8: Salary Hike and Performance ---
        st.header("Salary Hike and Performance")
        
        # 22. Performance Rating vs. Last Salary Hike Percent
        st.subheader("Performance Rating vs. Last Salary Hike Percent")
        st.markdown("Scatter plot with regression line showing the relationship between last salary hike percent and performance rating.")
        
        fig_perf_salary_hike_percent, ax_perf_salary_hike_percent = plt.subplots(figsize=(10, 6))
        sns.regplot(x='EmpLastSalaryHikePercent', y='PerformanceRating', data=df, ax=ax_perf_salary_hike_percent, color='darkred')
        ax_perf_salary_hike_percent.set_xlabel('Last Salary Hike Percent')
        ax_perf_salary_hike_percent.set_ylabel('Performance Rating')
        ax_perf_salary_hike_percent.set_title('Performance Rating vs. Last Salary Hike Percent')
        st.pyplot(fig_perf_salary_hike_percent)
        
        
        # --- Theme 9: Attrition Analysis ---
        st.header("Attrition Analysis")
        
        # 23. Performance Rating Distribution by Attrition
        st.subheader("Performance Rating Distribution by Attrition")
        st.markdown("Box plot showing the distribution of performance ratings for employees who have and have not attrited.")
        
        fig_perf_attrition_box, ax_perf_attrition_box = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Attrition', y='PerformanceRating', data=df, ax=ax_perf_attrition_box, palette="pastel")
        ax_perf_attrition_box.set_xlabel('Attrition')
        ax_perf_attrition_box.set_ylabel('Performance Rating')
        ax_perf_attrition_box.set_title('Performance Rating Distribution by Attrition')
        st.pyplot(fig_perf_attrition_box)
        
        
        # 24. Attrition Rate by Department
        st.subheader("Attrition Rate by Department")
        st.markdown("Bar chart showing the attrition rate in each department.")
        
        attrition_rate_dept = df.groupby('EmpDepartment')['Attrition'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
        
        fig_attrition_rate_dept, ax_attrition_rate_dept = plt.subplots(figsize=(10, 6))
        attrition_rate_dept.plot(kind='barh', ax=ax_attrition_rate_dept, color='coral')
        ax_attrition_rate_dept.set_xlabel('Attrition Rate')
        ax_attrition_rate_dept.set_ylabel('Department')
        ax_attrition_rate_dept.set_title('Attrition Rate by Department')
        st.pyplot(fig_attrition_rate_dept)
        
        
        # 25. Attrition Rate by Job Role
        st.subheader("Attrition Rate by Job Role")
        st.markdown("Bar chart showing the attrition rate for each job role.")
        
        attrition_rate_jobrole = df.groupby('EmpJobRole')['Attrition'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
        
        fig_attrition_rate_jobrole, ax_attrition_rate_jobrole = plt.subplots(figsize=(12, 8))
        attrition_rate_jobrole.plot(kind='barh', ax=ax_attrition_rate_jobrole, color='tomato')
        ax_attrition_rate_jobrole.set_xlabel('Attrition Rate')
        ax_attrition_rate_jobrole.set_ylabel('Job Role')
        ax_attrition_rate_jobrole.set_title('Attrition Rate by Job Role')
        st.pyplot(fig_attrition_rate_jobrole)        

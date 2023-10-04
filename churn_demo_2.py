#!/usr/bin/env python
# coding: utf-8

# ## Churn Prediction

# ### Importing necessary libraries


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import streamlit as st

# churn_data = pd.read_csv(r"D:/datasets/Churn/Churn_Modelling.csv")
churn_data = pd.read_csv(r"./Churn_Modelling.csv")

# churn_data


# ### Data Preparation
#

# In[3]:
st.title("Churn Propensity Score")
file = st.file_uploader("upload file for Churn data",
                        type=["csv", "xls", "xlsx", 'txt'],
                        accept_multiple_files=False)

if file is not None:
    generated_data = pd.read_csv(file)

    # Renaming columns
    churn_data.columns = churn_data.columns.str.lower()

    churn_data = churn_data.rename(columns={'exited': 'churn'})

    # Dropping unnecessary columns
    churn_data = churn_data.drop(['surname', 'rownumber', 'customerid'], axis=1)
    # Replace geo values

    indian_states = [
        "Andhra Pradesh", "Bihar", "Goa",
        "Gujarat", "Kerala",
        "Madhya Pradesh", "Maharashtra",
    ]

    random.shuffle(indian_states)
    new_geo = churn_data
    new_geo['geography'] = [random.choice(indian_states) for i in range(len(churn_data))]

    # missing values
    missing_count = churn_data.isna().sum()

    # mapping categorical variables
    gender_mapping = {'Male': 0, 'Female': 1}
    # Create a dictionary to map geography values to numerical values
    geography_mapping = {

        "Bihar": 1,
        "Goa": 2,
        "Gujarat": 3,
        "Kerala": 4,
        "Madhya Pradesh": 5,
        "Maharashtra": 6,
        "Andhra Pradesh": 7
    }

    churn_mapped_data = churn_data.copy()
    churn_mapped_data['gender'] = churn_mapped_data['gender'].map(gender_mapping)
    # Use the map function to apply the mapping to the 'geography' column
    churn_mapped_data['geography'] = churn_mapped_data['geography'].map(geography_mapping)
    # Transform the Categorical Variables: Creating Dummy Variables
    churn_mapped_data = pd.get_dummies(churn_mapped_data, columns=['geography'])

    # summary of data
    churn_data_summary = churn_data.describe()
    print(churn_data_summary)

    # ##### creation of tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Model Dev and Validation", "Scoring"])

    with tab1:
        # #############Calculation of churn rate for categorical variables
        # Gender
        # Calculate churn rates by gender
        gender_churn_rates = churn_data.groupby('gender')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            gender_churn_rates.reset_index(),
            x='gender',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='gender'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Gender')
        fig.update_yaxes(title_text='Churn Rate ')

        # Set the x-axis tickvals and ticktext for categorical labels
        fig.update_xaxes(
            tickvals=gender_churn_rates.reset_index()['gender'],  # Values for tick positions
            ticktext=['Female', 'Male'],  # Corresponding labels
        )
        st.plotly_chart(fig)

        print(gender_churn_rates)

        # Geography

        # Calculate churn rates by geography
        geography_churn_rates = churn_data.groupby('geography')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            geography_churn_rates.reset_index(),
            x='geography',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='geography'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Geography')
        fig.update_yaxes(title_text='Churn Rate ')

        # Set the x-axis tickvals and ticktext for categorical labels
        fig.update_xaxes(
            tickvals=geography_churn_rates.reset_index()['geography'],  # Values for tick positions
            ticktext=["Andhra Pradesh", "Bihar", "Goa",
                      "Gujarat", "Kerala",
                      "Madhya Pradesh", "Maharashtra"],  # Corresponding labels
        )

        st.plotly_chart(fig)
        print(geography_churn_rates)

        # Calculate churn rates by geography
        hascrcard_churn_rates = churn_data.groupby('hascrcard')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            hascrcard_churn_rates.reset_index(),
            x='hascrcard',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='hascrcard'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Geography')
        fig.update_yaxes(title_text='Churn Rate ')

        # Set the x-axis tickvals and ticktext for categorical labels
        fig.update_xaxes(
            tickvals=hascrcard_churn_rates.reset_index()['hascrcard'],  # Values for tick positions
            ticktext=["No", "Yes"],  # Corresponding labels

        )

        st.plotly_chart(fig)
        print(hascrcard_churn_rates)

        # Calculate churn rates by geography
        isactivemember_churn_rates = churn_data.groupby('isactivemember')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            isactivemember_churn_rates.reset_index(),
            x='isactivemember',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='isactivemember'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Geography')
        fig.update_yaxes(title_text='Churn Rate ')

        # Set the x-axis tickvals and ticktext for categorical labels
        fig.update_xaxes(
            tickvals=isactivemember_churn_rates.reset_index()['isactivemember'],  # Values for tick positions
            ticktext=["No", "Yes"],  # Corresponding labels

        )

        st.plotly_chart(fig)
        print(isactivemember_churn_rates)

        #
        # percentiles = [0, 1, 5, 10, 25, 30, 40, 50, 60, 75, 80, 90, 95, 99, 100]
        #
        # percentiles_array = np.array(percentiles) / 100
        # credit_percentile_values = churn_data['creditscore'].quantile(percentiles_array)
        # # Display the percentile values
        # print(f"\ncredit score percentile : \n{credit_percentile_values}")
        #
        #
        #
        # # Create a line chart using Plotly
        # fig = go.Figure(data=go.Scatter(x=percentiles_array, y=credit_percentile_values, mode='lines+markers'))
        #
        # # Customize the layout
        # fig.update_layout(
        #     title='Percentile Values Line Chart for Credit score',
        #     xaxis_title='Percentiles',
        #     yaxis_title='Values'
        # )
        #
        # st.plotly_chart(fig)
        #
        # # for age
        # age_percentile_values = churn_data['age'].quantile(percentiles_array)
        # # Display the percentile values
        # print(f"\nage percentile : \n{age_percentile_values}")
        #
        # # Create a line chart using Plotly
        # fig = go.Figure(data=go.Scatter(x=percentiles_array, y=age_percentile_values, mode='lines+markers'))
        #
        # # Customize the layout
        # fig.update_layout(
        #     title='Percentile Values Line Chart for age',
        #     xaxis_title='Percentiles',
        #     yaxis_title='Values'
        # )
        #
        # # Show the chart
        # st.plotly_chart(fig)
        #
        # # for tenure
        # tenure_percentile_values = churn_data['tenure'].quantile(percentiles_array)
        # # Display the percentile values
        # print(f"\ntenure percentile : \n{tenure_percentile_values}")
        #
        # # Create a line chart using Plotly
        # fig = go.Figure(data=go.Scatter(x=percentiles_array, y=tenure_percentile_values, mode='lines+markers'))
        #
        # # Customize the layout
        # fig.update_layout(
        #     title='Percentile Values Line Chart for Tenure',
        #     xaxis_title='Percentiles',
        #     yaxis_title='Values'
        # )
        #
        # # Show the chart
        # st.plotly_chart(fig)
        #
        # # for balance
        # balance_percentile_values = churn_data['balance'].quantile(percentiles_array)
        # # Display the percentile values
        # print(f"\nbalance percentile : \n{balance_percentile_values}")
        #
        # # Create a line chart using Plotly
        # fig = go.Figure(data=go.Scatter(x=percentiles_array, y=balance_percentile_values, mode='lines+markers'))
        #
        # # Customize the layout
        # fig.update_layout(
        #     title='Percentile Values Line Chart for balance',
        #     xaxis_title='Percentiles',
        #     yaxis_title='Values'
        # )
        #
        # # Show the chart
        # st.plotly_chart(fig)
        #
        # # for estimatedsalary
        # estimatedsalary_percentile_values = churn_data['estimatedsalary'].quantile(percentiles_array)
        # # Display the percentile values
        # print(f"\nestimated salary percentile : \n{estimatedsalary_percentile_values}")
        #
        # # Create a line chart using Plotly
        # fig = go.Figure(data=go.Scatter(x=percentiles_array, y=estimatedsalary_percentile_values, mode='lines+markers'))
        #
        # # Customize the layout
        # fig.update_layout(
        #     title='Percentile Values Line Chart for estimated salary',
        #     xaxis_title='Percentiles',
        #     yaxis_title='Values'
        # )
        #
        # # Show the chart
        # st.plotly_chart(fig)

        # #### Making Bins of the continous variables

        # making slabs/bins
        # Define the  slabs (bins),

        age_bins_1 = [18, 25, 35, 45, 55, 65, 75, 100]

        df_bin = churn_mapped_data.copy()

        # Create a new column 'Slab' with the  categories
        df_bin['age_slab'] = pd.cut(df_bin['age'], bins=age_bins_1, right=False).astype(str)

        # Display the DataFrame with  slabs
        print(f"\nData after binning : \n{df_bin}")

        # churn rate

        # Calculate churn rate for age slab
        churn_rate = df_bin.groupby('age_slab')['churn'].mean() * 100

        # Create a Plotly line chart
        fig1 = px.line(
            churn_rate.reset_index(),
            x='age_slab',
            y='churn',
            markers=True,  # Enable markers
            labels={'churn': 'churn Rate (%)'},
            title=f'age'
        )

        fig1.update_traces(line_color='blue')  # Set the line color
        fig1.update_xaxes(title_text='age_slab')
        fig1.update_yaxes(title_text='churn Rate (%)')
        # Set the y-axis range and tick interval
        fig1.update_yaxes(
            # range=[0, 100],  # Set the range from 0 to 100
            dtick=10  # Set the tick interval to 10
        )

        st.plotly_chart(fig1)

        # Salary vs churn
        salary_bins = [18, 10000, 60000, 110000, 130000, 150000, 170000, 210000]  # Salary categories
        df_bin = churn_mapped_data.copy()

        # Create a new column 'Slab' with the categories
        df_bin['est_salary_slab'] = pd.cut(df_bin['estimatedsalary'], bins=salary_bins, right=False)

        # Calculate churn rate for each salary bin
        salary_bins_churn_rate = df_bin.groupby('est_salary_slab')['churn'].mean() * 100
        salary_bins_churn_rate = pd.DataFrame(salary_bins_churn_rate)
        salary_bins_churn_rate = salary_bins_churn_rate.reset_index()

        # Assuming you have a DataFrame named salary_bins_churn_rate
        # Convert 'est_salary_slab' to category data type
        salary_bins_churn_rate['est_salary_slab'] = salary_bins_churn_rate['est_salary_slab'].astype('category')

        # Sort the data in ascending order
        salary_bins_churn_rate.sort_values('est_salary_slab', inplace=True)

        # Convert 'est_salary_slab' back to string or object data type
        salary_bins_churn_rate['est_salary_slab'] = salary_bins_churn_rate['est_salary_slab'].astype(str)

        # Display the DataFrame
        salary_bins_churn_rate = salary_bins_churn_rate.sort_index()
        salary_bins_churn_rate.head(15)

        # Create a line chart
        fig = px.line(salary_bins_churn_rate, x='est_salary_slab', y='churn', title='salary',
                      markers=True)

        # Customize the appearance (optional)
        fig.update_xaxes(title_text='est_salary_slab')
        fig.update_yaxes(title_text='churn Rate (%)')

        # Display the chart
        st.plotly_chart(fig)

        creditscore_bins = [350, 500, 750, 800, 900]  # Credit Score categories

        df_bin = churn_mapped_data.copy()

        # Create a new column 'Slab' with the  categories
        df_bin['creditscore_slab'] = pd.cut(df_bin['creditscore'], bins=creditscore_bins, right=False).astype(str)

        # churn rate

        # Calculate churn rate for creditscore slab
        churn_rate = df_bin.groupby('creditscore_slab')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            churn_rate.reset_index(),
            x='creditscore_slab',
            y='churn',
            markers=True,  # Enable markers
            labels={'churn': 'churn Rate (%)'},
            title=f'creditscore'
        )

        fig.update_traces(line_color='blue')  # Set the line color
        fig.update_xaxes(title_text='creditscore_slab')
        fig.update_yaxes(title_text='churn Rate (%)')
        st.plotly_chart(fig)

        #
        # # Correlation matrix
        #
        # import plotly.express as px
        # import streamlit as st
        #
        # # Select the columns of interest
        # selected_columns = ['creditscore', 'age', 'tenure', 'balance', 'estimatedsalary']
        #
        # selected_features = churn_mapped_data[selected_columns]
        #
        # # Compute pairwise correlations
        # correlation_matrix = selected_features.corr()
        # st.subheader('Correlation Matrix')
        # # Create a heatmap plot using Plotly
        # fig = px.imshow(correlation_matrix, color_continuous_scale='BlueRed', labels=dict( color="Correlation"),)
        #
        # # Increase figure size
        # fig.update_layout(width=900, height=800)
        #
        # # Manually specify annotations with font color
        # annotations = []
        # for i in range(len(selected_columns)):
        #     for j in range(len(selected_columns)):
        #         annotations.append(
        #             dict(
        #                 text=round(correlation_matrix.iloc[i, j], 2),
        #                 x=selected_columns[i],
        #                 y=selected_columns[j],
        #                 xref="x1",
        #                 yref="y1",
        #                 showarrow=False,
        #                 font=dict(size=18, color='white'),  # Set font size and color
        #             )
        #         )
        #
        # # Add annotations to the heatmap
        # fig.update_layout(annotations=annotations)
        # fig.update_xaxes(side="top", tickfont=dict(size=18, color='black'))
        # fig.update_yaxes(tickfont=dict(size=18, color='black'))
        #
        #
        # # Display the Plotly figure using Streamlit
        # st.plotly_chart(fig)
        #
        # # ##### VIF Values
        #
        #
        # from patsy.highlevel import dmatrices
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        #
        # # Assuming you have a DataFrame called churn_data
        # # Create the design matrix for linear regression
        # formula = 'churn ~ creditscore + age + tenure + balance + numofproducts + estimatedsalary'
        # y_intercept, X_intercept = dmatrices(formula, data=churn_mapped_data, return_type='dataframe')
        #
        # # Calculate VIF for each explanatory variable
        # vif = pd.DataFrame()
        # vif['VIF'] = [variance_inflation_factor(X_intercept.values, i) for i in range(X_intercept.shape[1])]
        # vif['variable'] = X_intercept.columns
        #
        # # Print the VIF values
        # st.subheader('Multicollinearity Diagnostic')
        # st.table(vif)

    with tab2:
        # ### Model Building

        # #### Logistic Regression

        def fit_and_estimate(X, y):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
            print(X_train.shape, X_test.shape)
            fit_and_estimate.X_train_data = X_train
            fit_and_estimate.y_train_data = y_train
            fit_and_estimate.X_test_data = X_test

            ###############################################################################################
            #########fitting the model

            # Create a logistic regression model
            log_reg_model = LogisticRegression()
            # Fit the model to the training data to estimate parameters
            log_reg_model.fit(X_train, y_train)

            fit_and_estimate.log_reg_model = log_reg_model
            # Make predictions on the test set
            y_pred_train = log_reg_model.predict(X_train)
            y_pred = log_reg_model.predict(X_test)

            # Evaluate the log_reg_model
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred)

            train_confusion = confusion_matrix(y_train, y_pred_train)
            test_confusion = confusion_matrix(y_test, y_pred)

            report = classification_report(y_test, y_pred)

            print(f'Train Accuracy: {train_accuracy}')
            print(f'Test Accuracy: {test_accuracy}')

            st.subheader(f'Train Accuracy: {train_accuracy}')
            st.subheader(f'Test Accuracy: {test_accuracy}')

            print(f'train confusion Matrix:\n{train_confusion}')
            print(f'test confusion Matrix:\n{test_confusion}')

            ConfusionMatrixDisplay.from_estimator(log_reg_model, X_train, y_train)
            plt.title('Train Confusion Matrix\n')
            st.pyplot(plt)

            ConfusionMatrixDisplay.from_estimator(log_reg_model, X_test, y_test)
            plt.title('Test Confusion Matrix\n')
            st.pyplot(plt)

            print(f'Classification Report:\n{report}')

            #######################################################################################
            # predicted proba
            features = X_train.columns.to_list()

            train_ks_table = X_train.copy()
            train_ks_table['lg_predicted_probability'] = log_reg_model.predict_proba(X_train[features])[:,
                                                         1]  # Predicted Proba for churn(=1)
            train_ks_table['churn'] = y_train  # Ground Truth
            fit_and_estimate.train_ks_table = train_ks_table

            test_ks_table = X_test.copy()
            test_ks_table['lg_predicted_probability'] = log_reg_model.predict_proba(X_test[features])[:,
                                                        1]  # Predicted Proba for churn(=1)
            test_ks_table['churn'] = y_test  # Ground Truth
            fit_and_estimate.test_ks_table = test_ks_table

            ########################################################################################
            # train roc curve

            y_score = log_reg_model.predict_proba(X_train)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_score)

            fig = px.area(
                x=fpr, y=tpr,
                title=f'Train: ROC Curve (AUC={auc(fpr, tpr):.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            st.plotly_chart(fig)

            # test roc curve

            y_score = log_reg_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            fig = px.area(
                x=fpr, y=tpr,
                title=f'Test: ROC Curve (AUC={auc(fpr, tpr):.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            st.plotly_chart(fig)

            ########################################################################################

            # Define and fit model
            sm_Log_model = sm.Logit(y_train, X_train).fit()

            log_reg_summary = sm_Log_model.summary()

            # Convert the summary to a DataFrame
            train_data_estimate = pd.DataFrame(log_reg_summary.tables[1])

            # Set the column names to the first row of the DataFrame
            train_data_estimate.columns = train_data_estimate.iloc[0]

            # Drop the first row (which contains the column names)
            train_data_estimate = train_data_estimate[1:]

            # Reset the index
            train_data_estimate = train_data_estimate.reset_index(drop=True)
            print(f"\n train data estimate:\n")
            # Summary of results
            print(sm_Log_model.summary())
            # st.subheader("Parameter estimate Table:\n")
            # st.table(train_data_estimate)

            ########################################################################################
            # Create a DataFrame to store the parameter estimates

            test_parameter_estimate_table = pd.DataFrame(
                {'Variable': X_test.columns, 'Coefficient': sm_Log_model.params.values})

            # Display the parameter estimate table
            print(f"\n test_parameter_estimate_table: \n{test_parameter_estimate_table}")

            ########################################################################################
            # variation between test and train estimates

            coefficients_train = log_reg_model.coef_[0]

            coefficients_train = log_reg_model.coef_[0]
            print('columns_mop')

            features = X_train.copy()

            geography_mapping = {

                "geography_1": "Bihar",
                "geography_2": "Goa",
                "geography_3": "Gujarat",
                "geography_4": "Kerala",
                "geography_5": "Madhya Pradesh",
                "geography_6": "Maharashtra",
                "geography_7": "Andhra Pradesh",
            }
            # Use the .rename() function to replace index labels
            features = features.rename(columns=geography_mapping)
            print(features.columns)

            coef_train_df = pd.DataFrame({'Variable': features.columns, 'Coefficient_Train': coefficients_train})
            print(coef_train_df)

            # Color function to set the bar color based on values
            def set_bar_color(val):
                return 'orange' if val <= 0 else 'green'

            coef_train_df['color'] = coef_train_df['Coefficient_Train'].apply(set_bar_color)

            fig = px.bar(
                coef_train_df,
                x='Coefficient_Train',
                y='Variable',
                orientation='h',
                title='Feature Weights',
                labels={'Coefficient_Train': 'Feature Weight'},
            )

            fig.update_traces(marker_color=coef_train_df['color'])  # Set bar colors based on the 'color' column

            # Customize the x-axis and y-axis labels
            fig.update_xaxes(title_text='Feature Weight')
            fig.update_yaxes(title_text='Feature')

            # Add data labels
            fig.update_traces(text=coef_train_df['Coefficient_Train'].apply(lambda x: f'{x:.4f}'), textposition='auto')

            st.plotly_chart(fig)
            fit_and_estimate.coef_train_df = coef_train_df

            coefficients_df = pd.DataFrame({
                'Variable': test_parameter_estimate_table['Variable'],  # Assuming variable names are the same
                'Coefficient_Test': test_parameter_estimate_table['Coefficient'],
                'Coefficient_Train': coef_train_df['Coefficient_Train']

            })

            coefficients_df['Coefficient_Difference'] = coefficients_df['Coefficient_Test'] - coefficients_df[
                'Coefficient_Train']
            coefficients_df['Coefficient_Variation'] = coefficients_df['Coefficient_Difference'] / coefficients_df[
                'Coefficient_Train']

            print(f"\nvariation between test and train estimates:\n {coefficients_df}")
            # st.subheader("Coefficient Variation:\n")
            # st.table(coefficients_df)
            fit_and_estimate.coefficients_df = coefficients_df
            fit_and_estimate.coefficient = train_data_estimate.iloc[:, 1]

            return train_data_estimate


        def fit_and_estimate2(df):

            df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['churn'])
            fit_and_estimate2.df_train = df_train
            fit_and_estimate2.df_test = df_test

            print(df_train['churn'].value_counts(normalize=True))

            print(df_test['churn'].value_counts(normalize=True))

            X_train = df_train.drop(['churn'], axis=1)  # Features
            y_train = df_train['churn']  # Target variable

            ###############################################################################################
            #########fitting the model

            # Create a logistic regression model
            log_reg_model = LogisticRegression()
            # Fit the model to the training data to estimate parameters
            log_reg_model.fit(X_train, y_train)
            X_test = df_test.drop(['churn'], axis=1)  # Features

            y_test = df_test['churn']  # Target variable

            fit_and_estimate2.log_reg_model = log_reg_model
            # Make predictions on the test set
            y_pred_train = log_reg_model.predict(X_train)
            y_pred = log_reg_model.predict(X_test)

            # Evaluate the log_reg_model
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred)

            train_confusion = confusion_matrix(y_train, y_pred_train)
            test_confusion = confusion_matrix(y_test, y_pred)

            report = classification_report(y_test, y_pred)

            print(f'Train Accuracy: {train_accuracy}')
            print(f'Test Accuracy: {test_accuracy}')

            st.subheader(f'Train Accuracy: {train_accuracy}')
            st.subheader(f'Test Accuracy: {test_accuracy}')

            print(f'train confusion Matrix:\n{train_confusion}')
            print(f'test confusion Matrix:\n{test_confusion}')

            ConfusionMatrixDisplay.from_estimator(log_reg_model, X_train, y_train)
            plt.title('Train Confusion Matrix\n')
            st.pyplot(plt)

            ConfusionMatrixDisplay.from_estimator(log_reg_model, X_test, y_test)
            plt.title('Test Confusion Matrix\n')
            st.pyplot(plt)

            print(f'Classification Report:\n{report}')

            ########################################################################################
            # train roc curve

            y_score = log_reg_model.predict_proba(X_train)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_score)

            fig = px.area(
                x=fpr, y=tpr,
                title=f'Train: ROC Curve (AUC={auc(fpr, tpr):.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            st.plotly_chart(fig)

            # test roc curve

            y_score = log_reg_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)

            fig = px.area(
                x=fpr, y=tpr,
                title=f'Test: ROC Curve (AUC={auc(fpr, tpr):.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain='domain')
            st.plotly_chart(fig)

            ########################################################################################

            # Define and fit model
            sm_Log_model = sm.Logit(y_train, X_train).fit()

            log_reg_summary = sm_Log_model.summary()

            # Convert the summary to a DataFrame
            train_data_estimate = pd.DataFrame(log_reg_summary.tables[1])

            # Set the column names to the first row of the DataFrame
            train_data_estimate.columns = train_data_estimate.iloc[0]

            # Drop the first row (which contains the column names)
            train_data_estimate = train_data_estimate[1:]

            # Reset the index
            train_data_estimate = train_data_estimate.reset_index(drop=True)
            print(f"\n train data estimate:\n")
            # Summary of results
            print(sm_Log_model.summary())
            # st.subheader("Parameter estimate Table:\n")
            # st.table(train_data_estimate)

            ########################################################################################
            # Create a DataFrame to store the parameter estimates

            test_parameter_estimate_table = pd.DataFrame(
                {'Variable': X_test.columns, 'Coefficient': sm_Log_model.params.values})

            # Display the parameter estimate table
            print(f"\n test_parameter_estimate_table: \n{test_parameter_estimate_table}")

            ########################################################################################
            # variation between test and train estimates

            coef_train_df = pd.DataFrame({'Variable': X_train.columns, 'Coefficient_Train': coefficients_train})
            print(coef_train_df)

            # Color function to set the bar color based on values
            def set_bar_color(val):
                return 'orange' if val <= 0 else 'green'

            coef_train_df['color'] = coef_train_df['Coefficient_Train'].apply(set_bar_color)

            fig = px.bar(
                coef_train_df,
                x='Coefficient_Train',
                y='Variable',
                orientation='h',
                title='Feature Weights',
                labels={'Coefficient_Train': 'Feature Weight'},
            )

            fig.update_traces(marker_color=coef_train_df['color'])  # Set bar colors based on the 'color' column

            # Customize the x-axis and y-axis labels
            fig.update_xaxes(title_text='Feature Weight')
            fig.update_yaxes(title_text='Feature')

            # Add data labels
            fig.update_traces(text=coef_train_df['Coefficient_Train'].apply(lambda x: f'{x:.4f}'), textposition='auto')

            st.plotly_chart(fig)
            fit_and_estimate2.coef_train_df = coef_train_df

            coefficients_df = pd.DataFrame({
                'Variable': test_parameter_estimate_table['Variable'],  # Assuming variable names are the same
                'Coefficient_Test': test_parameter_estimate_table['Coefficient'],
                'Coefficient_Train': coef_train_df['Coefficient_Train']

            })

            coefficients_df['Coefficient_Difference'] = coefficients_df['Coefficient_Test'] - coefficients_df[
                'Coefficient_Train']
            coefficients_df['Coefficient_Variation'] = coefficients_df['Coefficient_Difference'] / coefficients_df[
                'Coefficient_Train']

            print(f"\nvariation between test and train estimates:\n {coefficients_df}")
            # st.subheader("Coefficient Variation:\n")
            # st.table(coefficients_df)
            fit_and_estimate2.coefficients_df = coefficients_df
            fit_and_estimate2.coefficient = train_data_estimate.iloc[:, 1]

            return train_data_estimate


        ##### Fitting the model on raw variables
        # fit_df=churn_mapped_data.drop(['estimatedsalary', 'balance'], axis=1)
        # fit2 = fit_and_estimate2(fit_df)

        # ##KS using excel method :
        # # Probabilities_df

        # # p(Y)= 1/1+e^-(c1f+c2f...)
        # #   Score = c +c1f + c2f + c3f
        # coefficient = fit_and_estimate2.coefficient.astype(str).astype(float).iloc[:-1]

        # def calculate_score(row):
        #     # Exclude the 'churn' column when calculating the score
        #     return sum(coef * feature for coef, feature in zip(coefficient, row.drop('churn')))

        # # Apply the function to each row of the DataFrame
        # score_prob_df = fit_and_estimate2.df_train.copy()

        # # Calculate 'CxF' without multiplying the 'churn' column
        # score_prob_df['CxF'] = score_prob_df.apply(calculate_score, axis=1)

        # # Calculate 'X' and 'churn_prob' as before
        # score_prob_df['X'] = 1 + np.exp(-score_prob_df['CxF'])
        # score_prob_df['churn_prob'] = 1 / score_prob_df['X']

        # # Decile column

        # # Sort the DataFrame in ascending order
        # score_prob_df = score_prob_df.sort_values(by='churn_prob', ascending=False)
        # score_prob_df['rownumber'] = range(1, len(score_prob_df) + 1)

        # total_rows = len(score_prob_df)
        # n = 7
        # rows_per_decile = total_rows // n  # 7 deciles

        # # Create a list to hold decile values
        # deciles = []

        # # Assign decile values to the rows
        # for i in range(1, n + 1):  # n deciles
        #     start_index = (i - 1) * rows_per_decile
        #     end_index = i * rows_per_decile
        #     deciles.extend([i] * (end_index - start_index))

        # # Add the decile column to the DataFrame
        # score_prob_df['decile'] = deciles

        # #Lift df
        # # Group by 'decile' and calculate max and min probabilities
        # max_values = score_prob_df.groupby('decile')['churn_prob'].max()
        # min_values = score_prob_df.groupby('decile')['churn_prob'].min()
        # test_lift_df = pd.DataFrame(
        #     {'decile': max_values.index, 'max_value': max_values.values, 'min_value': min_values.values})
        # churn_count_df = score_prob_df.groupby('decile')['churn'].sum().reset_index()

        # # Rename the columns to match your 'result_df' format
        # churn_count_df.columns = ['decile', 'churn_count']

        # # Merge 'churn_count_df' with 'result_df' on the 'decile' column
        # test_lift_df = pd.merge(test_lift_df, churn_count_df, on='decile', how='left')
        # test_lift_df['non_churn_count'] = 1000 - test_lift_df['churn_count']

        # test_lift_df['cumulative_churn_count'] = test_lift_df['churn_count'].cumsum()
        # test_lift_df['cumulative_non_churn_count'] = test_lift_df['non_churn_count'].cumsum()
        # test_lift_df['%cumulative_churn'] = test_lift_df['cumulative_churn_count'] / test_lift_df['churn_count'].sum()
        # test_lift_df['%cumulative_non_churn'] = test_lift_df['cumulative_non_churn_count'] / test_lift_df['non_churn_count'].sum()
        # test_lift_df['KS'] = test_lift_df['%cumulative_churn'] - test_lift_df['%cumulative_non_churn']

        # st.subheader('Train Lift Table')
        # st.table(test_lift_df)

        # KC using Kaggle method
        X = churn_mapped_data.drop(['churn'], axis=1)  # Features
        y = churn_mapped_data['churn']  # Target variable
        X1 = X.drop(['estimatedsalary', 'balance'], axis=1)
        fit = fit_and_estimate(X1, y)
        print(fit)


        def calc_KS_chart(df_ks, df_type):
            # KS using Kaggle method
            KSdata = df_ks.copy()

            def calculate_ks_statistics(data, predicted_probability, ground_truth, response_name='churn'):
                # Sort the data in descending order of predicted probabilities.
                data = data.sort_values(by=predicted_probability, ascending=False)

                # Create deciles based on the predicted probabilities.
                # label_mapping = {'min','max'}

                data['decile_group'] = pd.qcut(data[predicted_probability], q=10)

                # Create columns for success and non-success responses.
                KS_data = data.groupby('decile_group').agg(
                    total_count=pd.NamedAgg(column=ground_truth, aggfunc='count'),
                    success_count=pd.NamedAgg(column=ground_truth, aggfunc='sum')
                ).sort_index(ascending=False)

                # Calculate additional statistics.
                KS_data['Number of Non-' + response_name] = KS_data['total_count'] - KS_data['success_count']
                KS_data[response_name + '_Rate (%)'] = (KS_data['success_count'] / KS_data['total_count'] * 100).round(
                    2)
                KS_data['Percent of ' + response_name + ' (%)'] = (
                        (KS_data['success_count'] / KS_data['success_count'].sum()) * 100).round(2)
                KS_data['Percent of Non-' + response_name + ' (%)'] = (
                            (KS_data['Number of Non-' + response_name] / KS_data[
                                'Number of Non-' + response_name].sum()) * 100).round(2)
                KS_data['ks_stats'] = ((KS_data['Percent of ' + response_name + ' (%)'].cumsum() - KS_data[
                    'Percent of Non-' + response_name + ' (%)'].cumsum()).round(4)).astype(float)

                KS_data['max_ks'] = np.where(KS_data['ks_stats'] == KS_data['ks_stats'].max(), 'Yes', '')

                # Calculate Gain and Lift.
                KS_data['Gain'] = KS_data['Percent of ' + response_name + ' (%)'].cumsum()
                KS_data['Lift'] = (KS_data['Gain'] / np.arange(10, 100 + 10, 10)).round(2)

                return KS_data

            # Example usage:
            ks_data = calculate_ks_statistics(KSdata, 'lg_predicted_probability', ground_truth='churn')

            st.subheader(f"{df_type} KS table")
            st.dataframe(ks_data)

            def model_selection_by_gain_chart(model_gains_dict):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(0, 100 + 10, 10)), y=list(range(0, 100 + 10, 10)),
                                         mode='lines+markers', name='Random Model'))
                for model_name, model_gains in model_gains_dict.items():
                    model_gains.insert(0, 0)
                    fig.add_trace(go.Scatter(x=list(range(0, 100 + 10, 10)), y=model_gains,
                                             mode='lines+markers', name=model_name))
                fig.update_xaxes(
                    title_text=f"% of {df_type} Data Set", )

                fig.update_yaxes(title_text="% of Gain", )
                fig.update_layout(title=f'{df_type} Gain Chart', )
                st.plotly_chart(fig)

            model_selection_by_gain_chart({'Log_regression': ks_data.Gain.to_list()})

            def model_selection_by_lift_chart(model_lift_dict):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(10, 100 + 10, 10)), y=np.repeat(1, 10),
                                         mode='lines+markers', name='Random Lift'))
                for model_name, model_lifts in model_lift_dict.items():
                    fig.add_trace(go.Scatter(x=list(range(10, 100 + 10, 10)), y=model_lifts,
                                             mode='lines+markers', name=model_name))
                fig.update_xaxes(
                    title_text=f"% of {df_type} Data Set", )

                fig.update_yaxes(title_text="Lift", )
                fig.update_layout(title=f'{df_type}  Lift Chart', )
                st.plotly_chart(fig)

            model_selection_by_lift_chart({'Log_regression': ks_data.Lift.to_list()})

            return ks_data


        train_ks = fit_and_estimate.train_ks_table
        train_ks_data = calc_KS_chart(train_ks, "Train")
        print('row count for first and last decile')
        print(train_ks[train_ks['lg_predicted_probability'] >= 0.417]['lg_predicted_probability'].count())
        print(train_ks[train_ks['lg_predicted_probability'] <= 0.0556]['lg_predicted_probability'].count())

        test_ks = fit_and_estimate.test_ks_table
        test_ks_data = calc_KS_chart(test_ks, "Test")


        def rm_insig_val(df, X_train, y_train):

            sig_features = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts',
                            'hascrcard', 'isactivemember', 'estimatedsalary', 'intercept']

            print(f"\n Significant Features : {sig_features} \n")

            while True:

                X_train_1 = X_train[sig_features]
                # Define and fit model

                model = sm.Logit(y_train, X_train_1).fit()

                # Summary of results
                print(f"train data estimate : {model.summary().tables[1]}")

                # Get the p-values for each feature (excluding the 'Intercept' column)
                p_values = model.pvalues

                # Find the features with p-values greater than 0.05
                insig_features = p_values[p_values > 0.05]
                print(f"\nfeatures to be removed:{insig_features.count()}")
                print(insig_features)

                if insig_features.empty:
                    break

                # Remove the feature(s) with high p-values
                feature_to_remove = insig_features.idxmax()

                print(f"\n Feature to remove: {feature_to_remove}")
                sig_features.remove(feature_to_remove)

            print(f"after removing all: {model.summary().tables[1]}")
            return sig_features

    with tab3:

        # ### Sample data generation

        # import pandas as pd
        # import random
        #
        # # Load your original data into a DataFrame (assuming it's in a CSV file)
        # original_data = churn_mapped_data.copy()
        #
        # # Create an empty DataFrame to store the generated data
        # generated_data = pd.DataFrame(columns=original_data.columns)
        #
        # # Number of rows you want to generate
        # num_rows_to_generate = 10000
        #
        # # Sample rows from your original data to create the new data
        # for _ in range(num_rows_to_generate):
        #     # Randomly select a row from the original data (with replacement)
        #     sampled_row = original_data.sample(n=1, replace=True)
        #
        #     # Append the sampled row to the generated data
        #     generated_data = pd.concat([generated_data, sampled_row])
        #     # generated_data = generated_data.append(sampled_row, ignore_index=True)

        # # Shuffle the 'Gender' column randomly
        # gender_values = generated_data['gender'].values
        # np.random.shuffle(gender_values)
        # geo_values = generated_data['geography'].values
        # np.random.shuffle(geo_values)
        # geo_values = generated_data['geography'].values
        # np.random.shuffle(geo_values)
        # balance_values = generated_data['balance'].values
        # np.random.shuffle(balance_values)
        #
        # # Assign the shuffled values back to the DataFrame
        # generated_data['gender'] = gender_values
        # generated_data['geography'] = geo_values
        # generated_data['balance'] = balance_values
        #
        #
        # # Generate random numbers (-2 or +4) for each row
        # credit_adjustor = np.random.choice([-2, 4], size=len(generated_data))
        #
        # # Update the 'creditscore' column by adding or subtracting the random numbers
        # generated_data['creditscore'] = generated_data['creditscore'] + credit_adjustor
        #
        #
        #
        #
        # # Save the generated data to a CSV file
        # generated_data.to_csv('D:/datasets/Churn/similar_generated_data.csv', index=False)
        #
        # # Display the first few rows of the generated data
        # generated_data.head()
        # # Convert specific columns to float
        # float_to_convert = ['balance', 'estimatedsalary']
        # generated_data[float_to_convert] = generated_data[float_to_convert].astype(float)
        #
        # int_to_convert = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'numofproducts', 'hascrcard', 'isactivemember', 'churn']
        # generated_data[int_to_convert] = generated_data[int_to_convert].astype(int)

        # In[19]:

        # save generated_data to  file
        # generated_data = pd.read_csv("D:/datasets/Churn/similar_generated_data.csv")
        # generated_data = pd.read_csv("similar_generated_data.csv")

        # #### Scoring

        # p(Y)= 1/1+e^-(c1f+c2f...)
        #   Score = c +c1f + c2f + c3f
        coefficient = fit_and_estimate.coefficient.astype(str).astype(float).iloc[:-1]


        def calculate_score(row):
            return sum(coef * feature for coef, feature in zip(coefficient, row))


        # Apply the function to each row of the DataFrame
        score_data = generated_data.copy()
        score_data = score_data.drop(['churn', 'estimatedsalary', 'balance'], axis=1)

        # score_data = score_data.drop('intercept', axis=1)

        score_data['CxF'] = score_data.apply(calculate_score, axis=1)

        #   X = 1 + e ^ -Score

        score_data['X'] = 1 + np.exp(-score_data['CxF'])

        #   P(Y) = 1 / X

        score_data['churn_prob'] = 1 / score_data['X']
        # IF(J2>0.7,"High",IF(J2>0.4,"Medium","Low"))
        score_data['churn_category'] = score_data['churn_prob'].apply(
            lambda x: "High" if x > 0.7 else ("Medium" if x > 0.4 else "Low"))

        score_data['churn_category'] = score_data['churn_prob'].apply(
            lambda x: "High" if x > 0.7 else ("Medium" if x > 0.4 else "Low"))

        print(set(score_data['churn_category'].to_list()))

        df = score_data.copy()
        df['customer_id'] = ['C' + str(i) for i in range(1, len(df) + 1)]
        df.set_index('customer_id', inplace=True)

        columns_to_drop = ['CxF', 'X']
        score_data_op = df.drop(columns=columns_to_drop)
        category_percentages = (score_data['churn_category'].value_counts(normalize=True) * 100).reset_index()
        category_percentages.columns = ['churn_category', 'percentage']  # Set column names

        # Define color mapping for categories
        color_mapping = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}

        category_order = ['High', 'Medium', 'Low']

        # Create a histogram using Plotly with category order and custom colors
        fig = px.bar(
            category_percentages,
            x='churn_category',
            y='percentage',  # Use 'percentage' for the y-axis
            title='Churn Prediction Category',
            labels={'churn_category': 'Churn Category', 'percentage': 'Churn Percentage'},
            color='churn_category',
            color_discrete_map=color_mapping,
            category_orders={'churn_category': category_order},
        )

        st.plotly_chart(fig)

    st.download_button(
        "Click to Download",
        pd.DataFrame(score_data_op).to_csv(),
        "scored_customers.csv",
        "text/csv",
        key='download-csv'
    )

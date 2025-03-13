#!/usr/bin/env python
# coding: utf-8

# ## Churn Prediction

# ### Importing necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import warnings
warnings.filterwarnings("ignore")

import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from yellowbrick.cluster import SilhouetteVisualizer
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import streamlit as st
from scipy.spatial.distance import cdist



churn_data = pd.read_csv(r"./Churn_Modelling.csv")
# churn_data = pd.read_csv(r"C:/Users/shreyasaraf/PycharmProjects/Churn Project/New folder/churn-pred/Churn_Modelling.csv")

#    churn_data


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

    # ##### creation of tabs.
    tab1, tab2, tab3 = st.tabs(["Scoring", "Model Dev and Validation", "EDA"])

    with tab3:
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
            title='Gender'
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
            title='Geography'
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

        # Calculate churn rates by hascrcard
        hascrcard_churn_rates = churn_data.groupby('hascrcard')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            hascrcard_churn_rates.reset_index(),
            x='hascrcard',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='Has credit card'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Has Credit Card')
        fig.update_yaxes(title_text='Churn Rate ')

        # Set the x-axis tickvals and ticktext for categorical labels
        fig.update_xaxes(
            tickvals=hascrcard_churn_rates.reset_index()['hascrcard'],  # Values for tick positions
            ticktext=["No", "Yes"],  # Corresponding labels

        )

        st.plotly_chart(fig)
        print(hascrcard_churn_rates)

        # Calculate churn rates by isactivemember
        isactivemember_churn_rates = churn_data.groupby('isactivemember')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            isactivemember_churn_rates.reset_index(),
            x='isactivemember',
            y='churn',
            markers=True,
            labels={'churn': 'Churn Rate '},
            title='Is active member'
        )

        fig.update_traces(line_color='blue')
        fig.update_xaxes(title_text='Is Active Member')
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
        age_churn_rate = df_bin.groupby('age_slab')['churn'].mean() * 100

        # Create a Plotly line chart
        fig1 = px.line(
            age_churn_rate.reset_index(),
            x='age_slab',
            y='churn',
            markers=True,  # Enable markers
            labels={'churn': 'churn Rate'},
            title=f'Age'
        )

        fig1.update_traces(line_color='blue')  # Set the line color
        fig1.update_xaxes(title_text='age_slab')
        fig1.update_yaxes(title_text='churn Rate')
        # Set the y-axis range and tick interval
        fig1.update_yaxes(
            # range=[0, 100],  # Set the range from 0 to 100
            dtick=10  # Set the tick interval to 10
        )

        st.plotly_chart(fig1)
        print(age_churn_rate)

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
        fig = px.line(salary_bins_churn_rate, x='est_salary_slab', y='churn', title='Salary',
                      markers=True)

        # Customize the appearance (optional)
        fig.update_xaxes(title_text='est_salary_slab')
        fig.update_yaxes(title_text='churn Rate')

        # Display the chart
        st.plotly_chart(fig)
        print(salary_bins_churn_rate)
      
        creditscore_bins = [350, 500, 750, 800, 900]  # Credit Score categories

        df_bin = churn_mapped_data.copy()

        # Create a new column 'Slab' with the  categories
        df_bin['creditscore_slab'] = pd.cut(df_bin['creditscore'], bins=creditscore_bins, right=False).astype(str)

        # churn rate

        # Calculate churn rate for creditscore slab
        creditscore_churn_rate = df_bin.groupby('creditscore_slab')['churn'].mean() * 100

        # Create a Plotly line chart
        fig = px.line(
            creditscore_churn_rate.reset_index(),
            x='creditscore_slab',
            y='churn',
            markers=True,  # Enable markers
            labels={'churn': 'churn Rate'},
            title=f'Credit score'
        )

        fig.update_traces(line_color='blue')  # Set the line color
        fig.update_xaxes(title_text='creditscore_slab')
        fig.update_yaxes(title_text='churn Rate (%)')
        st.plotly_chart(fig)
        print(creditscore_churn_rate)

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

        # #### Splitting Data

        X = churn_mapped_data.drop(['churn'], axis=1)  # Features
        y = churn_mapped_data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42,stratify =y)

        # Kmeans
        churn_df=churn_mapped_data.copy()
        continous_features = churn_df[['creditscore', 'age', 'numofproducts', 'estimatedsalary', 'tenure', 'balance']]

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        continous_features_scaled = scaler.fit_transform(continous_features)
        #continous_features_scaled[continous_features_scaled.columns] = StandardScaler().fit_transform(continous_features_scaled)
        continous_features_scaled = pd.DataFrame(continous_features_scaled, columns=continous_features.columns)

        print(continous_features_scaled.describe())

        
        kmeans = KMeans(2)

        kmeans.fit(X_train[continous_features_scaled.columns])


        identified_clusters = kmeans.fit_predict(X_train[continous_features_scaled.columns])
        


        X_train_with_clusters = X_train.copy()
        X_train_with_clusters['Clusters'] = identified_clusters 





        # Assuming you have a X_trainset called 'X_train' and you want to find the optimal k

        distortions = []
        K = list(range(1, 11))  # Convert the range to a list

        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(X_train[continous_features_scaled.columns])
            distortions.append(sum(np.min(cdist(X_train[continous_features_scaled.columns], kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_train[continous_features_scaled.columns].shape[0])



        # Create the elbow plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K, y=distortions, mode='lines+markers', name='Distortion'))
        fig.update_layout(
           
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Distortion'
        )
        st.subheader(f'Clustering')
        st.plotly_chart(fig)


        


        # Create a 2x3 subplot grid for 6 plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Iterate through the range of clusters and create SilhouetteVisualizer for each
        for n_clusters, ax in zip(range(2, 8), axes.ravel()):
            km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, random_state=42)
            visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax)
            visualizer.fit(X_train[continous_features_scaled.columns])
            ax.set_title(f"Silhouette Plot for {n_clusters} Clusters")

        plt.tight_layout()
        # plt.show()




        print(f"Silhouette Score for X_train")
        for i in range(2, 8):  # Change the range to cover 10 clusters
            km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
            km.fit(X_train[continous_features_scaled.columns])
            score = metrics.silhouette_score(X_train[continous_features_scaled.columns], km.labels_, metric='euclidean')
            
            print(f"Cluster : {i}")
            print(score)


        def canonical_plot(data):   
            clus_train = data[continous_features_scaled.columns].copy()


            # '''
            # K-MEANS ANALYSIS - INITIAL CLUSTER SET
            # '''
            # k-means cluster analysis for 1-10 clusters due to the 10 possible class outcomes for poker hands                                                       
            from scipy.spatial.distance import cdist
            clusters=range(1,11)
            meandist=[]
            # average distances from data points to their respective cluster centroids for each corresponding number of clusters.

            # loop through each cluster and fit the model to the train set
            # generate the predicted cluster assingment and append the mean distance my taking the sum divided by the shape
            

            # Interpret 2 cluster solution
            model3=KMeans(n_clusters=4)
            model3.fit(clus_train) # has cluster assingments based on using 2 clusters
            clusassign=model3.predict(clus_train)

            # plot clusters
            # ''' Canonical Discriminant Analysis for variable reduction:
            # 1. creates a smaller number of variables
            # 2. linear combination of clustering variables
            # 3. Canonical variables are ordered by proportion of variance accounted for
            # 4. most of the variance will be accounted for in the first few canonical variables
            # '''
            # from sklearn.decomposition import PCA # CA from PCA function
            pca_2 = PCA(2) # return 2 first canonical variables
            plot_columns = pca_2.fit_transform(clus_train) #reduce the data to two dimensions

            # """
            # BEGIN multiple steps to merge cluster assignment with clustering variables to examine
            # cluster variable means by cluster
            # """
            # # create a unique identifier variable from the index for the
            # cluster training data to merge with the cluster assignment variable
            clus_train.reset_index(level=0, inplace=True)
            # create a list that has the new index variable
            cluslist=list(clus_train['index'])
            # create a list of cluster assignments
            labels=list(model3.labels_)
            # combine index variable list with cluster assignment list into a dictionary
            newlist=dict(zip(cluslist, labels))
            
            # convert newlist dictionary to a dataframe
            newclus=pd.DataFrame.from_dict(newlist, orient='index')
            
            # rename the cluster assignment column
            newclus.columns = ['cluster']

            # now do the same for the cluster assignment variable create a unique identifier variable from the index for the
            # cluster assignment dataframe to merge with cluster training data
            newclus.reset_index(level=0, inplace=True)
            # merge the cluster assignment dataframe with the cluster training variable dataframe
            # by the index variable
            merged_train=pd.merge(clus_train, newclus, on='index')
            # cluster frequencies


            # """
            # END multiple steps to merge cluster assignment with clustering variables to examine
            # cluster variable means by cluster
            # """

            # FINALLY calculate clustering variable means by cluster
            canonical_plot.clustergrp = merged_train.groupby('cluster').mean()



            # PCA Scatterplot
            fig = px.scatter(clus_train, x=plot_columns[:, 0], y=plot_columns[:, 1], color=model3.labels_)
            fig.update_layout(
                xaxis_title='Canonical variable 1',
                yaxis_title='Canonical variable 2',
                title='Scatterplot of Canonical Variables for 4 Clusters'
            )
            st.plotly_chart(fig)
                    
        st.subheader(f"Train Clusters")
        can_train=canonical_plot(X_train)
        X_train_mean=canonical_plot.clustergrp
        st.subheader(f"Test Clusters")
        can_test=canonical_plot(X_test)
        X_test_mean=canonical_plot.clustergrp

        print(f"\n\nX_Train Mean: {X_train_mean}")
        print(f"\n\n\nX_Test Mean {X_test_mean}")

        # Isolation Forest
        IF_classifier = IsolationForest(n_estimators  = 2_000,
                                contamination = 0.01,
                                random_state  = 42)

        IF_classifier.fit(X_train)
        metrics_df=X_train.copy()
        pred = IF_classifier.predict(metrics_df)
        metrics_df['anomaly']=pred
        outliers=metrics_df.loc[metrics_df['anomaly']==-1]
        outlier_index=list(outliers.index)
        #print(outlier_index)
        #Find the number of anomalies and normal points here points classified -1 are anomalous
        print(metrics_df['anomaly'].value_counts())

        # Visualize Outliers
        #Train
        # Step 1: Perform PCA to reduce dimensionality
        n_components = 2  # You can adjust the number of components
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(metrics_df)

        # Step 2: Use the reduced data for anomaly detection
        IF_classifier = IsolationForest(n_estimators=2_000, contamination=0.01, random_state=42)
        IF_classifier.fit(X_pca)

        # Step 3: Identify and store the indices of outliers
        pred = IF_classifier.predict(X_pca)
        metrics_df['anomaly'] = pred
        outliers = metrics_df.loc[metrics_df['anomaly'] == -1]
        outlier_index = list(outliers.index)

        # Step 4: Plot the outliers using Plotly
        fig = go.Figure()

        # Plot normal data points
        normal_data = X_pca[metrics_df['anomaly'] == 1]
        fig.add_trace(go.Scatter(x=normal_data[:, 0], y=normal_data[:, 1], mode='markers', name='normal points'))

        # Plot outliers
        outlier_data = X_pca[metrics_df['anomaly'] == -1]
        fig.add_trace(go.Scatter(x=outlier_data[:, 0], y=outlier_data[:, 1], mode='markers', name='anomalies', marker=dict(color='red')))


        # Set layout properties
        fig.update_layout(
            title="Outliers Detected by Isolation Forest for Train  ",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2"
        )

        # Show the plot
        st.plotly_chart(fig)
        # Find the number of anomalies and normal points

        #Test
        IF_classifier = IsolationForest(n_estimators  = 2_000,
                                contamination = 0.01,
                                random_state  = 42)

        IF_classifier.fit(X_test)
        metrics_df=X_test.copy()
        pred = IF_classifier.predict(metrics_df)
        metrics_df['anomaly']=pred
        outliers=metrics_df.loc[metrics_df['anomaly']==-1]
        outlier_index=list(outliers.index)
        #print(outlier_index)
        #Find the number of anomalies and normal points here points classified -1 are anomalous
        # print(metrics_df['anomaly'].value_counts())

        # Step 1: Perform PCA to reduce dimensionality
        n_components = 2  # You can adjust the number of components
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(metrics_df)

        # Step 2: Use the reduced data for anomaly detection
        IF_classifier = IsolationForest(n_estimators=2_000, contamination=0.01, random_state=42)
        IF_classifier.fit(X_pca)

        # Step 3: Identify and store the indices of outliers
        pred = IF_classifier.predict(X_pca)
        metrics_df['anomaly'] = pred
        outliers = metrics_df.loc[metrics_df['anomaly'] == -1]
        outlier_index = list(outliers.index)

        # Step 4: Plot the outliers using Plotly
        fig = go.Figure()

        # Plot normal data points
        normal_data = X_pca[metrics_df['anomaly'] == 1]
        fig.add_trace(go.Scatter(x=normal_data[:, 0], y=normal_data[:, 1], mode='markers', name='normal points'))

        # Plot outliers
        # Plot outliers in red
        outlier_data = X_pca[metrics_df['anomaly'] == -1]
        fig.add_trace(go.Scatter(x=outlier_data[:, 0], y=outlier_data[:, 1], mode='markers', name='anomalies', marker=dict(color='red')))


        # Set layout properties
        fig.update_layout(
            title="Outliers Detected by Isolation Forest for Test  ",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2"
        )

        # Show the plot
        st.plotly_chart(fig)
        # Find the number of anomalies and normal points


        #Pipeline Estimates

                
        def estimate_pipeline(model, model_name):
            pipe = Pipeline([
                ('scaler', StandardScaler()),  # Step 1: Standardize the features
                ('selector', VarianceThreshold()),  # Step 2: Remove features with low variance
                ('classifier', model)  # Step 3: Random Forest Classifier (can be replaced with other classifiers)
            ])

            # X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=40)
                

            pipe.fit(X_train, y_train)

            training_score = pipe.score(X_train, y_train)
            test_score = pipe.score(X_test, y_test)
            
            # Create a dictionary to store the results
            results = {
                "Algorithm": model_name,
                "parameters_used":str(model),
                "Train Accuracy": training_score,
                "Test Accuracy": test_score
            }
            
            return results
            

        # Initialize a list to store results for different models
        results_list = []

        log_reg_results = estimate_pipeline(LogisticRegression(), "Logistic Regression")
        results_list.append(log_reg_results)

        random_forest_pipeline=estimate_pipeline(RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=50, n_jobs=-1, random_state=42), "Random Forest")
        results_list.append(random_forest_pipeline)

        svc_pipeline=estimate_pipeline(SVC(), "Support Vector Machine")
        results_list.append(svc_pipeline)

        naive_bayes_model = GaussianNB()
        NB_pipeline=estimate_pipeline(naive_bayes_model, "Naive Bayes model")
        results_list.append(NB_pipeline)

      
        gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gradient_boosting_results = estimate_pipeline(gradient_boosting_model, "Gradient Boosting")
        results_list.append(gradient_boosting_results)

        #NN Classifier
        NN_classifier = Sequential()
        NN_classifier.add(Dense(8, activation = 'relu',input_dim = 16))
        NN_classifier.add(Dense(8, activation = 'relu'))
        ### If we have more than 2 categories in the dependent variable then the activation function used in the output layer is softmax
        ### softmax is a sigmoid activation function for dependent variable with more than 2 categories
        ## and Output_dim = 3 for 3 categories in the dependent variable
        NN_classifier.add(Dense(1,activation = 'sigmoid'))

        NN_classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        history = NN_classifier.fit(X_train, y_train, batch_size=10, epochs=10)

        # Access the accuracy values from the history
        train_accuracy = history.history['accuracy']  # List of training accuracy values
        last_accuracy = train_accuracy[-1]


        # Create a dictionary to store the result
        result_dict={
            "Algorithm":"Simple Neural Network",
            "parameters_used":NN_classifier
        }
        result_dict["Train Accuracy"] = last_accuracy

        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        history = NN_classifier.fit(X_test,y_test,batch_size = 10, epochs = 10)
        test_accuracy = history.history['accuracy']  # List of training accuracy values

        last_accuracy = test_accuracy[-1]
        result_dict["Test Accuracy"] = last_accuracy

        results_list.append(result_dict)
        results_df = pd.DataFrame(results_list)
        results_df['Average Accuracy'] = (results_df['Train Accuracy'] + results_df['Test Accuracy']) / 2

        # results_df

        selected_columns = ['Algorithm', 'Train Accuracy', 'Test Accuracy']
        results_df_op = results_df[selected_columns]
        # Assuming your DataFrame is named 'df'
        results_df_op = results_df_op.sort_values(by=['Train Accuracy', 'Test Accuracy'], ascending=False)
        results_df_op = results_df_op.reset_index(drop=True)
        results_df_op.index = results_df_op.index + 1
        table_style = f"<style> table {{ font-size: 20px; }} </style>"

        st.subheader("Pipeline Estimates: ")

        st.write(table_style, unsafe_allow_html=True)
        st.table(results_df_op)

        # Select Best Model
        best_algorithm = results_df.loc[results_df['Average Accuracy'].idxmax()]
        best_algorithm_model = best_algorithm['parameters_used'].replace('\n', '')
        best_algorithm_model_name = best_algorithm['Algorithm'].replace('\n', '')
        st.write(f"Best Algorithm Model: {best_algorithm_model_name}")

        # Model Interpretations
        st.header('Model Interpretations')     
        model = eval(best_algorithm_model)
        model.fit(X_train, y_train)

        #accuracies
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        st.subheader('Train  Accuracy score  : {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
        st.subheader('Test  Accuracy score  : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

        y_true = y_test  # Replace with your actual labels

        precision = precision_score(y_true, y_pred_test)
        recall = recall_score(y_true, y_pred_test)
        f1 = f1_score(y_true, y_pred_test)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

        #Confusion Matrix
                
        train_confusion = confusion_matrix(y_train, y_pred_train)
        test_confusion = confusion_matrix(y_test, y_pred_test)


        print(f'train confusion Matrix:\n{train_confusion}')
        print(f'test confusion Matrix:\n{test_confusion}')



        def plot_swapped_columns_transposed_conf_matrix(conf, dftype):
            import plotly.figure_factory as ff
            import numpy as np

            # Swap columns in the confusion matrix
            swapped_columns_conf_matrix = np.array(conf)[:, [1, 0]].tolist()

            # Transpose the swapped columns matrix
            transposed_swapped_columns_conf_matrix = np.array(swapped_columns_conf_matrix).tolist()



            # Define colors for font based on TP, TN, FP, FN
            font_colors = [['red', 'green'],
                            ['green', 'red']]

            # Create custom annotation text with font colors for the transposed matrix
            annotations = []
            for i in range(2):
                for j in range(2):
                    label = 'TP' if i ==  1 and j == 0 else 'FP' if i == 0 and j == 0 else 'TN' if i == 0 and j == 1 else 'FN'
                    value = transposed_swapped_columns_conf_matrix[i][j]
                    font_color = font_colors[i][j]
                    annotations.append(dict(
                        x=j,
                        y=i,
                        text=f'{value}<br>{label}',
                        showarrow=False,
                        font=dict(color=font_color)
                    ))

            

            # Create a figure with custom annotations and color scale for the transposed matrix
            fig = ff.create_annotated_heatmap(
                z=transposed_swapped_columns_conf_matrix,
                x=['Predicted Positive', 'Predicted Negative'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale=[[0, 'beige'], [1, '#94CCFB']],
                showscale=False,  # No color scale
                annotation_text=transposed_swapped_columns_conf_matrix,
                customdata=transposed_swapped_columns_conf_matrix,
            )

            # Add custom annotations to the figure
            fig.update_layout(annotations=annotations)

            # Customize the layout
            fig.update_layout(
                
                xaxis=dict(title='Predicted'),
                yaxis=dict(title='Actual'),
            )
            st.subheader(f'{dftype} Confusion Matrix: \n')
            # Show the plot
            st.plotly_chart(fig)
        # Example usage
        plot_swapped_columns_transposed_conf_matrix(train_confusion, "Train")
        plot_swapped_columns_transposed_conf_matrix(test_confusion, "Test")

        #K fold Cross Validation
        

        def calc_Kfold(model):
            # Define the number of folds (k) for cross-validation
            k = 10  # You can adjust this value


            st.subheader(f"K Fold Cross Validation for Random Forest model\n")
            # Perform k-fold cross-validation
            cross_val_scores = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')


            "Train K Fold cross validation"
            fold_accuracies = []

            # Print the accuracy scores for each fold
            for fold, accuracy in enumerate(cross_val_scores, start=1):
                 fold_accuracies.append(f'Fold {fold}: Accuracy = {accuracy:.2f}')
            
            # Group the fold accuracies in pairs for each row
            fold_rows = [fold_accuracies[i:i+2] for i in range(0, len(fold_accuracies), 2)]

            # Join the fold accuracy pairs in each row with spaces
            for row in fold_rows:
                st.write('    '.join(row))

            # Calculate and print the mean accuracy
            mean_accuracy = np.mean(cross_val_scores)
            f'Train Mean Accuracy: {mean_accuracy:.2f}\n\n'

            cross_val_scores = cross_val_score(model, X_test, y_test, cv=k, scoring='accuracy')


            "Test K Fold cross validation"
            fold_accuracies = []

            # Print the accuracy scores for each fold
            for fold, accuracy in enumerate(cross_val_scores, start=1):
                 fold_accuracies.append(f'Fold {fold}: Accuracy = {accuracy:.2f}')

            # Group the fold accuracies in pairs for each row
            fold_rows = [fold_accuracies[i:i+2] for i in range(0, len(fold_accuracies), 2)]

            # Join the fold accuracy pairs in each row with spaces
            for row in fold_rows:
                st.write('    '.join(row))
            # Calculate and print the mean accuracy
            mean_accuracy = np.mean(cross_val_scores)
            f'Test Mean Accuracy: {mean_accuracy:.2f}\n\n'




        random_forest_pipeline=calc_Kfold(RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=50,
                            n_jobs=-1, random_state=42))


        # ROC Curve
        y_prob_train = model.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, y_prob_train)
        roc_auc = auc(fpr, tpr)

        train_roc_curve_fig = go.Figure()

        # Add the ROC curve trace
        train_roc_curve_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc),
                        legendgroup="group"  # this can be any string, not just "group")
                        ))

        # Add a diagonal line (random classifier)
        train_roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Classifier'))

        # Customize the layout
        train_roc_curve_fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=10, r=10, t=30, b=10),
            autosize=True,

        )
        
        st.subheader("Train ROC Curve")
        # Show the ROC curve
        st.plotly_chart(train_roc_curve_fig)


        # Assuming you have a trained classifier rf_classifier and test data X_test, y_test
        y_prob_test = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
        roc_auc = auc(fpr, tpr)

        test_roc_curve_fig = go.Figure()

        # Add the ROC curve trace
        test_roc_curve_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC = {:.2f})'.format(roc_auc))
        )

        # Add a diagonal line (random classifier)
        test_roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Classifier'))

        # Customize the layout
        test_roc_curve_fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate (FPR)',
            yaxis_title='True Positive Rate (TPR)',
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=10, r=10, t=30, b=10),
            autosize=True,
        )
        st.subheader("Test ROC Curve")

        # Show the ROC curve
        st.plotly_chart(test_roc_curve_fig)

        #Feature Importance Plot
        
        # Get feature importances
        feature_importances = model.feature_importances_
    
        geography_mapping = {

            "geography_1": "Bihar",
            "geography_2": "Goa",
            "geography_3": "Gujarat",
            "geography_4": "Kerala",
            "geography_5": "Madhya Pradesh",
            "geography_6": "Maharashtra",
            "geography_7": "Andhra Pradesh",
            "isactivemember": "is active member",
            "hascrcard":"has credit ard",
            "numofproducts": "num of products"            

        }
        features = X_train.copy()

        # Use the .rename() function to replace index labels
        features = features.rename(columns=geography_mapping)
        print(features.columns)



        # Assuming you have a list of feature names
        # feature_names = X_train.columns

        # Create a DataFrame with feature names and their importances
        feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})

        # Sort the DataFrame by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Create the Plotly figure for feature importances
        fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Random Forest Feature Importance',
            labels={'Importance': 'Feature Importance'},
        )

        # Set the y-axis order to be descending based on Importance
        fig.update_yaxes(categoryorder='total ascending')

        # Customize the x-axis and y-axis labels
        fig.update_xaxes(title_text='Feature Importance')
        fig.update_yaxes(title_text='Feature')

        # Set a different color for positive and negative importances (optional)
        fig.update_traces(marker_color=['green' if imp > 0 else 'red' for imp in feature_importance_df['Importance']])
        # Add data labels with increased font size
        fig.update_traces(
            text=feature_importance_df['Importance'].apply(lambda x: f'{x:.3f}'),  # Format to three decimals
            textposition='outside',
            textfont=dict(size=20)  # Set the desired font size here
        )

        # Show the Plotly figure
        st.plotly_chart(fig)


        











        

    with tab1:

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

        score_data1 = generated_data.copy()
        score_data1['gender'] = score_data1['gender'].map(gender_mapping)
        # Use the map function to apply the mapping to the 'geography' column
        score_data1['geography'] = score_data1['geography'].map(geography_mapping)
        # Transform the Categorical Variables: Creating Dummy Variables
        score_data1 = pd.get_dummies(score_data1, columns=['geography'])

        geography_mapping = {

            "geography_1": "Bihar",
            "geography_2": "Goa",
            "geography_3": "Gujarat",
            "geography_4": "Kerala",
            "geography_5": "Madhya Pradesh",
            "geography_6": "Maharashtra",
            "geography_7": "Andhra Pradesh",
            "isactivemember": "is active member",
            "hascrcard":"has credit card",
            "numofproducts": "num of products",

        }


        cust_id_index=score_data1['customer_id']
        score_data1 = score_data1.drop(['customer_id'], axis=1)
        prob= model.predict_proba(score_data1)[:, 1]

        # Use the .rename() function to replace index labels
        score_data1 = score_data1.rename(columns=geography_mapping)


        score_data1['probability']=prob

        #add categories
        score_data1['churn_category'] = score_data1['probability'].apply(lambda x: "High" if x > 0.7 else ("Medium" if x > 0.4 else "Low"))


                
        print(set(score_data1['churn_category'].to_list()))

        category_percentages = (score_data1['churn_category'].value_counts(normalize=True) * 100).reset_index()
        category_percentages.columns = ['churn_category', 'percentage']  # Set column names
        score_data1.insert(0, 'customer_id', cust_id_index)


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
        # Add data labels to the bars
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='auto')

        st.plotly_chart(fig)

        # Merge 'score_data' columns into 'generated_data' based on 'customer_id'
        final_op=generated_data = generated_data.merge(score_data1[['customer_id','probability', 'churn_category']], on='customer_id', how='left')
        final_op.set_index('customer_id', inplace=True)


    st.download_button(
        "Click to Download",
        pd.DataFrame(final_op).sort_values(by = 'probability', ascending = False).to_csv(),
        "scored_customers.csv",
        "text/csv",
        key='download-csv'
    )

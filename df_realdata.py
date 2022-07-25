import pandas as pd
import plotly.express as px
import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, array, random, argsort
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


st.title("Audit app") 
st.markdown('Use this Streamlit app to detect outliers') 

df4 = st.file_uploader('Select Your Local CSV for outlier analysis') 
if df4 is not None: 
	df = pd.read_csv(df4) 
else: 
	st.stop()

    
owners = st.sidebar.multiselect('Main column Filter', df['Country of Dealer'].unique()) 

if owners: 
    df = df[df['Country of Dealer'].isin(owners)]  
#perc_heads = st.selectbox('What do want the x variable to be?', ['CAD', 'USD']) 
#st.sidebar.header ("Использование фильтров в данных. Выберите показатели")
#year = st.sidebar.multiselect ( "Выберите год:", options = df["GJAHR"].unique(), default = df["GJAHR"].unique())
#df_selection = df.query (" WAERS == @owners")
    #st.dataframe (df)
    
    # Use the amount column and scale it for analysis
    X = df[['Equipment Cost (USD)']]
    # X = df[['DMBTR','BUKRS','LIFNR']] # Additional features can be added this way
    X = pd.DataFrame(X)
    X = scale(X)

    # Train the K-Means model by fitting it with the data. This is a one cluster model because it is used to detect anomalies for the amount feature.
    kmeans = KMeans(n_clusters = 1).fit(X)

    # Get the center of the cluster
    center = kmeans.cluster_centers_

    # Calculate the euclidean distance of a given data point from the center of the cluster
    distance = sqrt((X - center)**2)

    # Sort the distance along with the index of the data points
    order_index = argsort(distance, axis = 0)

    # Get the three farthest points
    indexes = order_index[-3:]
    values = X[indexes]

    # Plot the result. Here the red points indicate the suspected anomalies based on their distance from the other points.
    #x_ax = range(X.shape[0])
    #plt.plot(x_ax, X)
    #plt.scatter(indexes, values, color='r')
    #plt.show()

    col1, col2 = st.columns(2) 

    with col1: 
        x_ax = range(X.shape[0])
        fig1, ax1 = plt.subplots() 
        ax = plt.plot(x_ax, X)
        plt.scatter(indexes, values, color='r')
        st.pyplot(fig1) 

    with col2:


        x_ax = range(50)
        fig2, ax2 = plt.subplots() 
        sns.distplot(df['Equipment Cost (USD)'], bins=10)
        plt.xticks(rotation = 90)
        plt.xlabel('Index')
        plt.ylabel('Amount ($)')
        st.pyplot(fig2) 
        #x_ax = range(X.shape[0])
        #fig2, ax2 = plt.subplots() 
        #ax = plt.plot(x_ax, X)
        #plt.scatter(indexes, values, color='g')
        #st.pyplot(fig2) 
    
    indexes3 =list(indexes.flatten())
    df5 = df.iloc[indexes3]
    st.dataframe (df5)
else:

    #st.dataframe (df)
    
    # Use the amount column and scale it for analysis
    X = df[['Equipment Cost (USD)']]
    # X = df[['DMBTR','BUKRS','LIFNR']] # Additional features can be added this way
    X = pd.DataFrame(X)
    X = scale(X)

    # Train the K-Means model by fitting it with the data. This is a one cluster model because it is used to detect anomalies for the amount feature.
    kmeans = KMeans(n_clusters = 1).fit(X)

    # Get the center of the cluster
    center = kmeans.cluster_centers_

    # Calculate the euclidean distance of a given data point from the center of the cluster
    distance = sqrt((X - center)**2)

    # Sort the distance along with the index of the data points
    order_index = argsort(distance, axis = 0)

    # Get the three farthest points
    indexes = order_index[-3:]
    values = X[indexes]

    # Plot the result. Here the red points indicate the suspected anomalies based on their distance from the other points.
    #x_ax = range(X.shape[0])
    #plt.plot(x_ax, X)
    #plt.scatter(indexes, values, color='r')
    #plt.show()

    col1, col2 = st.columns(2) 

    with col1: 
        x_ax = range(X.shape[0])
        fig1, ax1 = plt.subplots() 
        ax = plt.plot(x_ax, X)
        plt.scatter(indexes, values, color='r')
        st.pyplot(fig1) 

    with col2:


        x_ax = range(50)
        fig2, ax2 = plt.subplots() 
        sns.distplot(df['Equipment Cost (USD)'], bins=10)
        plt.xticks(rotation = 90)
        plt.xlabel('Index')
        plt.ylabel('Amount ($)')
        st.pyplot(fig2) 
        #x_ax = range(X.shape[0])
        #fig2, ax2 = plt.subplots() 
        #ax = plt.plot(x_ax, X)
        #plt.scatter(indexes, values, color='g')
        #st.pyplot(fig2) 
    
    indexes3 =list(indexes.flatten())
    df5 = df.iloc[indexes3]
    st.dataframe (df5)
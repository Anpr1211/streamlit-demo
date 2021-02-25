# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:21:22 2021

@author: ankita
"""
# importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA                     # principal component analysis
import plotly.express as px 
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# reading the dataset
data = pd.read_csv("https://raw.githubusercontent.com/Anpr1211/streamlit-demo/main/test.csv", index_col='Unnamed: 0', nrows=1000)

data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction'])

data.drop(['id', 'Gender_Female', 'Customer Type_disloyal Customer', 'Type of Travel_Business travel', 'satisfaction_neutral or dissatisfied',
           'Arrival Delay in Minutes'],
          axis=1, inplace=True)

st.subheader("Displaying the dataset")
st.dataframe(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# creating and fitting the model
model = kmeans(n_clusters=2, n_init=25).fit(scaled_data)

# initialising the PCA
pca = PCA(n_components=2)

# fitting the PCA
principalComponents = pca.fit_transform(scaled_data)

# making a dataframe of the principal components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_component_1', 'principal_component_2'])

st.subheader("Displaying the Principal Components of the dataset")
st.dataframe(principalDf)

fig = px.scatter(data_frame=principalDf, x='principal_component_1', y='principal_component_2', color=model.labels_)

st.subheader("Displaying the clusters")
st.plotly_chart(fig)

# list to store the within sum of squared error for the different clusters given the respective cluster size
wss = []

# loop to iterate over the no. of clusters and calculate the wss
for i in range(1,11):
    # kmeans
    fitx = kmeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(scaled_data)
    # appending the value
    wss.append(fitx.inertia_)

# plot
fig_elbow, ax = plt.subplots(figsize=(11,8.5))
ax.plot(range(1,11), wss, 'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal $k$')

st.subheader("The Elbow Curve")
st.pyplot(fig_elbow)

st.header("User Interaction")
input = st.text_area("Input from the user", value='0.81788702, -1.03517064,  1.70385282,  0.62164094,  0.17214266, 0.79783121, -0.1617392 ,  0.54469211, -0.34030327,  1.22713822, 1.25917014,  1.25097391,  1.16171635, -1.03534761,  1.14406253, 1.29899809,  0.95381034, -0.98593202,  0.47603968, -0.66337845, -0.9627357 ,  1.11636981, -0.28227491,  1.1304853')
input_clean = [float(i.strip()) for i in input.split(", ")]

output = model.predict(np.array(input_clean).reshape(1, 24))

st.text("The closest cluster of the input data is {}.".format(output[0]))


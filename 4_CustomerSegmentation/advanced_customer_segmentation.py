# ADVANCED CUSTOMER SEGMENTATION
# In this project, we will use an advanced library (Kmodes) developed by the Massachusetts Institute of Technology (MIT).
# Since our data is complex, we cannot use K-Means here.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# The kmodes library is not included in the standard distribution, you may need to install it.
# It is an advanced library developed by the Massachusetts Institute of Technology (MIT).
# Details are available at https://anaconda.org/conda-forge/kmodes
# You can install it on Anaconda with the command # conda install -c conda-forge kmodes.
from kmodes.kprototypes import KPrototypes  
# You can download dataset from: https://www.kaggle.com/khalidnasereddin/retail-dataset-analysis?select=segmentation-data.csv#
df = pd.read_csv("segmentation-data.csv")
print('df head:',df.head())
# Let's look the dataset if we have any null?
print('null check:',df.isnull().sum())
# We have no null data, ok..
# Income ve Age Data Normalization
# Before Scaling/Normalization we keep our normal values in temp variables..
df_temp = df[['ID','Age', 'Income']]
print('df_temp:',df_temp)
# Scaling process..
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])
# Since ID is not used in analysis, Drop ID before analysis..
df = df.drop(['ID'], axis=1)
# Convert Age and Income into float..
mark_array= df.values
mark_array[:, 2] = mark_array[:, 2].astype(float)
mark_array[:, 4] = mark_array[:, 4].astype(float)
print('df head:',df.head())
# We are building our model...
kproto = KPrototypes(n_clusters=10, verbose=2, max_iter=20)
# kproto = KPrototypes(n_clusters=5, verbose=2, max_iter=20)
# kproto = KPrototypes(n_clusters=15, verbose=2, max_iter=20)
clusters = kproto.fit_predict(mark_array, categorical=[0, 1, 3, 5, 6])
print('centroids:',kproto.cluster_centroids_)
print('length:',len(kproto.cluster_centroids_))
cluster_col=[]
for c in clusters:
    cluster_col.append(c)
df['cluster']=cluster_col
# Put original columns from temp to df:
df[['ID','Age', 'Income']] = df_temp
# After clustering, you can now easily get lists of your customers for each cluster or segment:
print('df df cluster head:\n',df[df['cluster'] == 9].head(10))
colors = ['green', 'red', 'gray', 'orange', 'yellow', 'cyan', 'magenta', 'brown', 'purple', 'blue']
plt.figure(figsize=(15,15))
plt.xlabel('Age')
plt.ylabel('Income')
for i, col in zip(range(10), colors):
    dftemp = df[df.cluster==i]
    plt.scatter(dftemp.Age, dftemp['Income'], color=col, alpha=0.5)
plt.legend()
plt.show()
# Homework
# Change cluster size (using values both lower and greater than 10) and run the program again and see if you can get a better segmentation...

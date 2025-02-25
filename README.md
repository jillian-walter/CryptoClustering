# CryptoClustering
Module 19 Challenge | Penn Data Bootcamp

This Challenge details the process of using Unsupervised Machine Learning to evaluate Cryptocurrency price changes during 24-hour and 7-day periods to identify any clusters or patterns.
The data is provided in a CSV format (available in the /Resources folder), detailing % changes during different time periods.

Unsupervised Machine Learning differs from Supervised Machine Learning as it does not provide the machine with an output (Y) - only X variables. Unsupervised Machine Learning tells the machine to figure out the structure of the dataset. 

Libraries used for this Challenge include: 
- import pandas as pd
- import hvplot.pandas
- from sklearn.cluster import KMeans
- from sklearn.decomposition import PCA
- from sklearn.preprocessing import StandardScaler

# Previewing and Prepping the Data:

We begin by reading in the data from a CSV and converting it to a Dataframe using the Pandas library, we preview the data using **.head(), .describe() and hvplot** plotting the different cryptocurrencies on the HVPlot. With this, we can see that there are 41 rows in the dataset, and that the currencies Ethlend, Celcius-Degree-Token and Theta-Token are the most volatile.

To prep, we use the **StandardScaler().fit_transform** function to transform the different datapoints into the same scale --> (x-mu)/standard deviation.

# K-Means Clustering: 

The K-Means clustering process is always initiated by the Elbow Method, which is used to determine the optimal number of clusters for the dataset. Looping through a list of values 1-11, we use the fuction **.inertia_** to find the intertia values, and will plot the values using hvplot along a curved line. With the elbow curve method, you are looking for the "elbow" of the curve - the point at which the inertia slows and the line begins to plateau. Based on the chart, the "elbow" of the curve happens at cluster 4, which is where the inertia of the curve starts to plateau. It is worth also testing 5 clusters with other methods, such as the Silhouette Method or Calinski Harabasz Score to confirm that 4 is the optimal number of clusters.

We then initialize the K-Means model, plugging in 4 clusters. For this analysis, we used a random state of 0. 
![image](https://github.com/user-attachments/assets/50389828-b6f9-4fe5-91f8-939a82a271a7)

The next step is to use the **.predict** to create cluster predictions for each of the values, and then copy these cluster predictions into the dataframe with complete scaled data. Using hvplot, we are able to see the 4 distinct clusters mapped across 24-hour changes and 7-day changes.

![image](https://github.com/user-attachments/assets/6bc555d9-b159-4d81-a41c-aab637d26ea1)


# Principal Component Analysis:

Principal Component Analysis is used to remove the "noise" of multiple variables that can sometimes occur with K-Means clustering, by identifying the variables that best describe the dataset. 
We first identify the number of components (in this case, 3), and then use the **pca.fit_transform** function to apply the PCA variables to the existing scaled data. 
Using the formula **pca.explained_variance_ratio_** we can see that 37% of the data is explained by the first PCA variable, 35% is explained by the second PCA variable, and 18% explained by the third PCA variable. 

The same process as the K-Means Clustering method above is used to determine the PCA clusters. We first identify the optimal number of clusters using the elbow method (also 4), intiate the K-Means model, predict the clusters and plot using HV Plot. 

![image](https://github.com/user-attachments/assets/8abd88b7-081e-4011-9ae0-e31fadd95e5d)


# Findings:
Using fewer features allowed us to better isolate the differences amongst the four clusters. This is especially prevalent for cluster 3, which initially was indecipherable amongst cluster 0 data, while now, we can identify it as having a high values across both PCA 1 and PCA 2. The individual datapoints in each cluster are also now closer together & are closer to their centroids, while in the initial scaled chart the datapoints were more spread out. We can get an overall more cohesive & streamlined story using the PCA method, allowing us to develop cluster summaries (below):

- Cluster 0: These datapoints have both a low PCA1 value and a low PCA2 value
- Cluster 1: This datapoint sits in its own cluster, as it has a significantly high PCA 1 value but significantly low PCA2 value
- Cluster 2: These datapoints also have low PCA1 values (similar to cluster 0) but have higher PCA2 values
- Cluster 3: This datapoint sits in its own cluster, as it has both a high PCA1 value and a high PCA2 value

![image](https://github.com/user-attachments/assets/274b9278-fb2d-403a-bcdd-7258865fff19)


# Importing Modules
from os.path import sep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

p = ".." + sep + ".." + sep + "Participants" + sep + "OneMin" + sep + "Filtered" + sep + "CompleteFeatures" + sep + "CompleteFeatures.csv"
data = pd.read_csv(p)
data = data.iloc[:, 1:]
X = data.values

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

y_kmeans1 = y_kmeans
y_kmeans1 = y_kmeans + 1

cluster = pd.DataFrame(y_kmeans1)  # Adding cluster to the Dataset1
data['cluster'] = cluster  # Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(data.groupby('cluster').mean(), 1))
print(kmeans_mean_cluster.head())

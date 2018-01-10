# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np
from IPython import embed
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


# In[2]:


def load_prepared_data():
	df_train = pd.read_csv('./data/output/processed_train.csv', header=0)
	df_test = pd.read_csv('./data/output/processed_test.csv', header=0)
	features = list(set(df_train.columns) - {'Vote'})
	target = 'Vote'

	df_train_X = df_train[features]
	df_train_Y = df_train[target]
	df_test_X = df_test[features]
	df_test_Y = df_test[target]
	# labels = {"0":"Blues","1":"Browns","2":"Greens","3":"Greys","4":"Oranges","5":"Pinks","6":"Purples","7":"Reds","8":"Whites","9":"Yellows" }
	labels = ["Blues", "Browns", "Greens", "Greys", "Oranges", "Pinks", "Purples", "Reds", "Whites", "Yellows"]
	return df_train_X, df_train_Y, df_test_X, df_test_Y, labels


# In[15]:


df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()
df_train_Y = df_train_Y.map(lambda x: labels[int(x)])
df_test_Y = df_test_Y.map(lambda x: labels[int(x)])

train_val_data = pd.concat([df_train_X])
features = train_val_data.values
labels = pd.concat([df_train_Y]).values

# In[18]:


from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from time import time


def loop_bench_k_means(data):
	print(82 * '_')
	print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
	for i in range(2, 10):
		km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1)
		bench_k_means(km, "k-means++ k=" + str(i), data)



	# km = KMeans(n_clusters=i, init='random', max_iter=100, n_init=1)
	# bench_k_means(km, "random k=" + str(i), data)


def bench_k_means(estimator, name, data):
	t0 = time()
	estimator.fit(data)
	print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
	      % (name, (time() - t0), estimator.inertia_,
	         metrics.homogeneity_score(labels, estimator.labels_),
	         metrics.completeness_score(labels, estimator.labels_),
	         metrics.v_measure_score(labels, estimator.labels_),
	         metrics.adjusted_rand_score(labels, estimator.labels_),
	         metrics.adjusted_mutual_info_score(labels, estimator.labels_),
	         metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=1000)))


# km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
# km.fit(features)





# In[19]:

cluster_size = 4
instances = len(features)
loop_bench_k_means(features)

for cluster_size in range(2, 12):
	print(82 * '_')
	print "cluster_size %d" % cluster_size
	print(82 * '_')

	km = KMeans(n_clusters=cluster_size)
	km.fit(features)

	df_temp = df_train_Y.copy().to_frame()
	df_temp['cluster'] = km.labels_

	for i in range(0, cluster_size):
		my_df = df_temp[df_temp['cluster'] == i]
		c = Counter(my_df['Vote'].values)
		print "C ID %d, Total: %d, percent: %.2f%%, C: %s" % (i, len(my_df), float(len(my_df)) / instances * 100, c)








# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(features, km.labels_, sample_size=1000))


# coding: utf-8

# In[149]:


import pandas as pd
import matplotlib.pyplot as plt
import pylab as p

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
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
from sklearn import metrics




# In[150]:


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


# In[151]:


df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()
df_train_Y_nums = df_train_Y
df_test_Y_nums = df_test_Y
df_train_Y = df_train_Y.map(lambda x: labels[int(x)])
df_test_Y = df_test_Y.map(lambda x: labels[int(x)])

train_val_data = pd.concat([df_train_X])
features = train_val_data.values
labels = pd.concat([df_train_Y]).values


# In[152]:


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


# In[153]:


def loop_bench_k_means(data, num):
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    for i in range(2, num):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=1)
        bench_k_means(km, "k-means++ k=" + str(i), data)


# In[176]:


def k_means(num, X, Y):
    instances = len(X)
    loop_bench_k_means(X, num)

    for cluster_size in range(2, num):
        print(82 * '_')
        print "cluster_size %d" % cluster_size
        print(82 * '_')

        km = KMeans(n_clusters=cluster_size)
        km.fit(X)

        df_temp = Y.copy().to_frame()
        df_temp['cluster'] = km.labels_

        for i in range(0, cluster_size):
            my_df = df_temp[df_temp['cluster'] == i]
            c = Counter(my_df['Vote'].values)
            print "C ID %d, Total: %d, percent: %.2f%%, C: %s" % (i + 2, len(my_df), float(len(my_df)) / instances * 100, c)


# In[155]:


coalition = ['Purples', 'Browns', 'Greens', 'Pinks', 'Whites']
non_coalition = ['Greys', 'Oranges', 'Reds', 'Yellows', 'Blues']


# In[156]:


def transform_category(_df, name):
    for cat in _df[name].unique():
        _df[cat] = (_df[name] == cat).astype(int)
    return _df


# In[157]:


my_features = ['Overall_happiness_score', 'Garden_sqr_meter_per_person_in_residancy_area','Yearly_IncomeK']



# In[158]:


for party in non_coalition:
    df_party = df_train_X.dropna().copy()
    df_party['Vote'] = df_train_Y
    df_party = df_party[df_party['Vote'] == party]
    df_party[my_features].plot(kind='kde') 
    p.title(party)
    p.show()
    print df_party[my_features].mean()


# In[159]:


for party in coalition:
    df_party = df_train_X.dropna().copy()
    df_party['Vote'] = df_train_Y
    df_party = df_party[df_party['Vote'] == party]
    df_party[my_features].plot(kind='kde') 
    p.title(party)
    p.show()
    print df_party[my_features].mean()
    


# In[160]:


df_party = df_train_X.dropna().copy()
df_party['Vote'] = df_train_Y
df_party = df_party[df_party['Vote'].isin(coalition)]
df_party[my_features].plot(kind='kde') 
p.title(" Coalition")
p.show()

print df_party[my_features].mean()

df_party = df_train_X.dropna().copy()
df_party['Vote'] = df_train_Y
df_party = df_party[df_party['Vote'].isin(non_coalition)]
df_party[my_features].plot(kind='kde') 
p.title(" Non Coalition")
p.show()

print df_party[my_features].mean()
    


# In[161]:


pecentile = 80
from operator import itemgetter

def most_important_features_per_party(num):
    target = pd.concat([df_train_Y]).values
    df = df_train_Y.copy().to_frame()

    df = transform_category(df, 'Vote')
    df = df.drop('Vote', axis=1)

    print '_' * 82
    res = {}
    
    for party in df.columns.values:
        # clf_tree = DecisionTreeClassifier(min_samples_split=5, random_state=0)
        clf_tree = DecisionTreeClassifier()
        clf_tree.fit(features, df[party])
        imp = clf_tree.feature_importances_
        selected_columns = []

        for i in range(len(train_val_data.columns)):
            selected_columns.append([train_val_data.columns[i], imp[i]])        
        selected_columns = sorted(selected_columns, key=itemgetter(1), reverse=True)[:num]
        res[party] = selected_columns
    return res



# In[162]:


def most_important_features_overall():
    target = pd.concat([df_train_Y]).values
    df = df_train_Y.copy().to_frame()

    df = transform_category(df, 'Vote')
    df = df.drop('Vote', axis=1)

    selected_columns = []

    for party in df.columns.values:
        clf_tree = DecisionTreeClassifier(min_samples_split=5, random_state=0)
        clf_tree.fit(features, df[party])
        imp = clf_tree.feature_importances_
        result = imp > np.percentile(imp, pecentile)

        for i in range(len(result)):
            if result[i]:
                selected_columns.append(train_val_data.columns[i])

    print '_' * 82
    print Counter(selected_columns)


# In[163]:


def most_important_features_per_coalition():
    target = pd.concat([df_train_Y]).values
    df = df_train_Y.copy().to_frame()

    df = transform_category(df, 'Vote')
    df = df.drop('Vote', axis=1)
    
    coalition_columns = []
    non_coalition_columns = []

    for party in df.columns.values:
        clf_tree = DecisionTreeClassifier(min_samples_split=5, random_state=0)
        clf_tree.fit(features, df[party])
        imp = clf_tree.feature_importances_
        result = imp > np.percentile(imp, pecentile)

        for i in range(len(result)):
            if result[i]:
                if party in coalition:
                    coalition_columns.append(train_val_data.columns[i])
                else:
                    non_coalition_columns.append(train_val_data.columns[i])

    print '_' * 82
    print "Coalition", Counter(coalition_columns)
    print '_' * 82
    print "Non Coalition", Counter(non_coalition_columns)
    print '_' * 82


# In[177]:


k_means(25, df_train_X, df_train_Y)
# most_important_features_overall()
# most_important_features_per_coalition()


# In[165]:


most_important_features_per_party(3)


# In[166]:


def show_feature_party(feature, party):
    df_party = df_train_X.dropna().copy()
    df_party['Vote'] = df_train_Y
    df_party = df_party[df_party['Vote'] == party]
    df_party[feature].plot(kind='kde') 
    p.title(party + " " + feature)
    p.show()

    print df_party[feature].describe()


    


# In[167]:


show_feature_party('Garden_sqr_meter_per_person_in_residancy_area', 'Purples')   
show_feature_party('Garden_sqr_meter_per_person_in_residancy_area', 'Pinks')   


# In[168]:


show_feature_party('Overall_happiness_score', 'Reds')   
show_feature_party('Overall_happiness_score', 'Greys')
show_feature_party('Overall_happiness_score', 'Greens')
show_feature_party('Overall_happiness_score', 'Browns')




# In[174]:


def x():
    df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()

    train_val_data = pd.concat([df_train_X])
    features = train_val_data.values
    target = pd.concat([df_train_Y]).values
    
    features_test = df_test_X
    target_test = df_test_Y
    
    clf = DecisionTreeClassifier(min_samples_split=8)
    clf.fit(features, target)
    embed()
    pred = clf.predict(features_test)

    distribution = np.bincount(pred.astype('int64'))
    most_common = np.argmax(distribution)

    print "winner is party ## %s ##" % labels[most_common.astype('int')]

    print "Vote distribution"
    distribution = np.bincount(pred.astype('int64'))

    for index,party in enumerate(distribution):
        print "%s, %f, %f"%(labels[index], distribution[index], distribution[index]/ float(target_test.size) * 100) + '%'
x()



# coding: utf-8

# In[1]:


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
from operator import itemgetter



# # Loading the Data

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


# # Preparing the Data

# In[3]:


df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()
df_train_Y_nums = df_train_Y
df_test_Y_nums = df_test_Y
df_train_Y = df_train_Y.map(lambda x: labels[int(x)])
df_test_Y = df_test_Y.map(lambda x: labels[int(x)])

train_val_data = pd.concat([df_train_X])
features = train_val_data.values
labels = pd.concat([df_train_Y]).values


# ### Define Max K for out election - K = 25

# In[4]:


max_K = 26


# ### Becnhmarking K-Means on out data

# In[5]:


def print_bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.calinski_harabaz_score(data, estimator.labels_) ,
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=1000))
         )


# In[6]:


def loop_print_bench_k_means(data, labels, num):
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tcalinski\tsilhouette\t')
    for i in range(2, num):
        km = KMeans(n_clusters=i)
        print_bench_k_means(km, "k-means k=" + str(i), data, labels)
        
loop_print_bench_k_means(df_train_X, df_train_Y, max_K)


# In[127]:


def result_bench_k_means(data, labels, num):
    homogeneity_score = []
    completeness_score = []
    v_measure_score = []
    adjusted_rand_score = []
    adjusted_mutual_info_score = []
    silhouette_score = []
    calinski_harabaz_score = []
    interia = []


        
    for i in range(2, num):
        inner_homogeneity_score = []
        inner_completeness_score = []
        inner_v_measure_score = []
        inner_adjusted_rand_score = []
        inner_adjusted_mutual_info_score = []
        inner_silhouette_score = []
        inner_calinski_harabaz_score = []
        inner_interia = []
        
        k_fold = RepeatedStratifiedKFold(n_splits=2)
        for train_idx, test_idx in k_fold.split(data, labels):
#             print type(data)
#             print data[[0,3,5]]
#             print train_idx, len(data)
#             print data[train_idx]
            estimator = KMeans(n_clusters=i, random_state=0)
            estimator.fit(data.values[train_idx])
            inner_interia.append(estimator.inertia_)
            inner_homogeneity_score.append(metrics.homogeneity_score(labels[train_idx], estimator.labels_))
            inner_completeness_score.append(metrics.completeness_score(labels[train_idx], estimator.labels_))
            inner_v_measure_score.append(metrics.v_measure_score(labels[train_idx], estimator.labels_))
            inner_adjusted_rand_score.append(metrics.adjusted_rand_score(labels[train_idx], estimator.labels_))
            inner_adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(labels[train_idx], estimator.labels_))
            inner_silhouette_score.append(metrics.silhouette_score(data.values[train_idx], estimator.labels_, metric='euclidean', sample_size=1000))
            inner_calinski_harabaz_score.append(metrics.calinski_harabaz_score(data.values[train_idx], estimator.labels_)) ,
        
        interia.append(np.mean(inner_interia))
        homogeneity_score.append(np.mean(inner_homogeneity_score))
        completeness_score.append(np.mean(inner_completeness_score))
        v_measure_score.append(np.mean(inner_v_measure_score))
        adjusted_rand_score.append(np.mean(inner_adjusted_rand_score))
        adjusted_mutual_info_score.append(np.mean(inner_adjusted_mutual_info_score))
        silhouette_score.append(np.mean(inner_silhouette_score))
        calinski_harabaz_score.append(np.mean(inner_calinski_harabaz_score))
        
    return {
        "interia": interia,
        "homogeneity_score": homogeneity_score,
        "completeness_score": completeness_score,
        "v_measure_score" : v_measure_score,
        "adjusted_rand_score" :adjusted_rand_score,
        "adjusted_mutual_info_score": adjusted_mutual_info_score,
        "silhouette_score" : silhouette_score,
        "calinski_harabaz_score": calinski_harabaz_score
    }


# In[128]:



def draw_metrics(X, Y, k_max, inline=False):
    res = result_bench_k_means(X, Y, k_max)

    plot_x = range(2,k_max)
    
    for _metric, values in enumerate(res):
        plt.plot(plot_x, res[values], '8-')
        plt.xlabel('K')
        plt.ylabel(values)
        plt.title(values + ' measure on kMeans')
        plt.grid(True)

        plt.show()

        
draw_metrics(df_train_X, df_train_Y, max_K)



# ### We Chose K = 10

# In[10]:


k = 10


# In[11]:


km = KMeans(n_clusters=k, verbose=0, random_state=0)
print "Training: K=%d" % k
km.fit(df_train_X)
print "Done"


# In[12]:


def k_means_party_distribution(clf, X, Y, k):
    df_Y = Y.copy().to_frame()
    df_Y['cluster'] = clf.labels_
        
    res = {}
    for i in range(0, k):
        my_df = df_Y[df_Y['cluster'] == i]
        c = Counter(my_df['Vote'].values)
        res[i] = c
    return res


# In[13]:


def basic_distribution(dist):
    for key,val in dist.iteritems():
        items = val.most_common()
        keys = []
        values = []
        for item in items:
            keys.append(item[0])
            values.append(item[0])

        print "Group: %s, Distribution: %s"%(str(key), sorted(keys))


# ### Distrebution per cluster

# In[14]:


dist_per_cluster = k_means_party_distribution(km, df_train_X, df_train_Y, k)
basic_distribution(dist_per_cluster)


# ### Distrebution per party

# In[15]:


def k_means_cluster_distribution(clf, X, Y):

    df_Y = Y.copy().to_frame()
    df_Y['cluster'] = clf.labels_
        
    res = {}
    
    for i in labels:
        my_df = df_Y[df_Y['Vote'] == i]
        c = Counter(my_df['cluster'].values)
        res[i] = c
    return res


dist_per_party = k_means_cluster_distribution(km ,df_train_X, df_train_Y)
basic_distribution(dist_per_party)


# ### Summery of Distribution

# In[16]:


print ("_" *22) + "dist per cluster" + ("_" *22)
print "_" * 60
for i in dist_per_cluster:
    print i,dist_per_cluster[i]
    
print 
print ("_" *22) + " dist per party " + ("_" *22)
print "_" * 60    
for i in dist_per_party:
    print i,dist_per_party[i]


# ### Distances Between cluster centers

# In[94]:


from sklearn.metrics.pairwise import euclidean_distances

def build_distance_matrix(clf):
    m = euclidean_distances(clf.cluster_centers_, clf.cluster_centers_)
    m = np.around(m, decimals=3)
    
    matrix = np.matrix(m)
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    return matrix
    
matrix = build_distance_matrix(km)
for i in matrix:
    print i


print "Mean distance: %f" %matrix.mean()


# ### Trying GMM to see distances

# In[96]:


from sklearn import mixture
from sklearn.metrics.pairwise import euclidean_distances

gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit(df_train_X)

def build_distance_matrix_gmm(gmm):
    m = euclidean_distances(gmm.means_, gmm.means_)
    m = np.around(m, decimals=3)
    

    matrix = np.matrix(m)
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    return matrix

gmm_dist = build_distance_matrix_gmm(gmm)
print gmm_dist

print "Mean distance: %f" %gmm_dist.mean()


# Mean distance between clusters in gmm is smaller than in Kmeans
# 
# We continue with kmeans since it is more stable (distances are bigger)

# ### Grouping Clusters to Coalition
# in this step we look for the closest cluster and try to merge them if they have a matching parties.

# In[19]:


def build_candidates_for_merge(matrix):
    med = matrix.mean()
    pairs = []

    for i, line in enumerate(np.asarray(matrix)):
        for j, col in enumerate(line):
            if i < j and col < med:
                pairs.append((i,j, col))
            
    return  sorted(pairs, key=itemgetter(2))


pairs = build_candidates_for_merge(matrix)
for i in pairs:
    print i[0], sorted(dist_per_cluster[i[0]].keys())
    print i[1], sorted(dist_per_cluster[i[1]].keys())
    print "Clusters: %d --> %d, distance: %.03f"%i
    print "_" * 50


# ### Chosen Coalition
# This is the Chosen coalition

# In[20]:


coalition = ['Purples', 'Browns', 'Greens', 'Pinks', 'Whites']
non_coalition = ['Greys', 'Oranges', 'Reds', 'Yellows', 'Blues']
coalition_clusters = [0,1,3,4,6]


# ## Validating Coalition size

# In[21]:


def in_coal(row):
    if row["Vote"] in coalition:
        return 1
    else:
        return 0

def count_coalition(X, Y):
    df = Y.copy().to_frame()
    
    df['coal'] = df.apply (lambda row: in_coal(row),axis=1)
    val_counts = df['coal'].value_counts()
    print "In Coualtion the are %d votes which are %.02f%% percent"%(val_counts[1], float(val_counts[1]) / len(Y) * 100)


    
count_coalition(df_train_X, df_train_Y) 


# ## Validating Coalition size on test
# 

# In[22]:


def predict_coalition(train_X, train_Y, test_X, test_Y):
    print "Training..."
    clf = RandomForestClassifier(min_samples_split=4, random_state=0)
    clf.fit(train_X, train_Y)
    print "Pridicting..."
    pred = clf.predict(test_X)
    distribution = Counter(pred)

    print "predicted winner is party ## %s ##" % distribution.most_common(1)[0][0]
    df_pred = pd.DataFrame()
    df_pred['Vote'] = pred
    count_coalition(test_X, df_pred['Vote'])
    return df_pred

def predict_clustering(clf, train_X, train_Y, test_X, test_Y):
    cluster_pred = clf.predict(test_X)
    return cluster_pred

def count_coalition_clusters(pred):
    c = Counter(pred)
    in_coal = sum(c[cid] for cid in coalition_clusters)
    total = sum(c.values())  
    print "Clustering - In Coualtion the are %d votes which are %.02f%% percent"%(in_coal, float(in_coal) / total * 100)

    
prediction = predict_coalition(df_train_X, df_train_Y, df_test_X,df_test_Y)    
clustering_prediction = predict_clustering(km, df_train_X, df_train_Y, df_test_X, df_test_Y)    
count_coalition_clusters(clustering_prediction)


# # Important Features

# In[26]:


def transform_category(_df, name):
    for cat in _df[name].unique():
        _df[cat] = (_df[name] == cat).astype(int)
    return _df

def most_important_features_per_party(X, Y, num):
    target = pd.concat([Y]).values
    df = Y.copy().to_frame()

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

important_features = most_important_features_per_party(df_train_X, df_train_Y, 5)
for party in important_features:
    print party, important_features[party]
    
    


# In[27]:


def draw_most_important_features_per_party(party_feature_data):
#     party_feature_data = most_important_features_per_party(num)
    for party, values in party_feature_data.iteritems():
        keys = []
        vals = []
        for i in values:
            keys.append(i[0])
            vals.append(i[1])
        
        y_pos = np.arange(len(keys))
        plt.barh(y_pos, vals, align='center', alpha=0.5)
        plt.yticks(y_pos, keys)
        plt.xlim((0,1))
        plt.title(party)
        plt.xlabel('Most important features')

        plt.show()

draw_most_important_features_per_party(important_features)       
        


# In[28]:


# important_features
def feature_score_limit(important_features, min_score, max_score=1):
    for party in important_features:
        for features in important_features[party]:
            if features[1] > min_score and features[1] <= max_score:
                print party, features

print 0.5                
feature_score_limit(important_features, 0.5)  

print "0.4 - 0.5"
feature_score_limit(important_features, 0.4, 0.5)  

print "0.3 - 0.4"
feature_score_limit(important_features, 0.3, 0.4)  

print "0.25 - 0.3"
feature_score_limit(important_features, 0.25, 0.3)  



# ### Define most important features

# In[29]:


improtant_features = ['Overall_happiness_score', 
                      'Avg_Satisfaction_with_previous_vote',
                      'Garden_sqr_meter_per_person_in_residancy_area',
                      'Yearly_IncomeK',
                    'Number_of_valued_Kneset_members',
                     'Avg_monthly_expense_when_under_age_21']



# In[30]:


def plot_feature_distriution(parties, my_features, X, Y):
    for party in parties:
        df_party = X.copy()
        df_party['Vote'] = Y
        df_party = df_party[df_party['Vote'] == party]
        df_party = df_party[my_features]
        df_party.plot(kind='kde') 
        p.title(party)
        p.grid(True)
        p.show()
        print df_party[my_features].mean()
        
print "Coalition"
print "_" * 83
plot_feature_distriution(coalition, improtant_features, df_train_X, df_train_Y)
print "Non Coalition"
print "_" * 83
plot_feature_distriution(non_coalition, improtant_features, df_train_X, df_train_Y)


# In[31]:


def show_feature_party(feature, party):
    df_party = df_train_X.dropna().copy()
    df_party['Vote'] = df_train_Y
    df_party = df_party[df_party['Vote'] == party]
    df_party[feature].plot(kind='kde')
    p.xlim(0,1)
    p.title(party + " " + feature)
    p.show()

    print df_party[feature].describe()


# ### Plottig distributions

# In[32]:



show_feature_party('Overall_happiness_score', 'Pinks')
show_feature_party('Overall_happiness_score', 'Purples')




# In[33]:


show_feature_party('Overall_happiness_score', 'Reds')   
show_feature_party('Overall_happiness_score', 'Greys')
show_feature_party('Overall_happiness_score', 'Greens')
show_feature_party('Overall_happiness_score', 'Browns')
show_feature_party('Overall_happiness_score', 'Pinks')
show_feature_party('Overall_happiness_score', 'Oranges')


# In[34]:


show_feature_party('Garden_sqr_meter_per_person_in_residancy_area', 'Purples')   
show_feature_party('Garden_sqr_meter_per_person_in_residancy_area', 'Pinks')
show_feature_party('Garden_sqr_meter_per_person_in_residancy_area', 'Greens')   


# In[35]:


show_feature_party('Number_of_valued_Kneset_members', 'Purples')   
show_feature_party('Number_of_valued_Kneset_members', 'Pinks')


# Choosing a new winner by manipulation of data

# In[36]:



def pick_a_new_winner(train_X, train_Y, test_X, test_Y):
    clf = RandomForestClassifier(min_samples_split=4, random_state=0)
    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)    
    distribution = Counter(pred)

    print "Original winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print ""    

    print "Garden_sqr_meter_per_person_in_residancy_area = 0.33"
    new_test_X = test_X.copy()
    new_test_X['Garden_sqr_meter_per_person_in_residancy_area'] = 0.33
    
    pred = clf.predict(new_test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print ""
    
    print "Will_vote_only_large_party = 0.9"
    new_test_X = test_X.copy()
    new_test_X['Will_vote_only_large_party'] = 0.9
    
    
    pred = clf.predict(new_test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print ""
    
    print "Number_of_valued_Kneset_members = 0.236734"
    new_test_X = test_X.copy()
    new_test_X['Number_of_valued_Kneset_members'] = 0.236734

    
    pred = clf.predict(new_test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print ""
    
    print "Garden_sqr_meter_per_person_in_residancy_area /= 0.236734"
    print "Number_of_valued_Kneset_members -= 0.23"
    new_test_X = test_X.copy()
    new_test_X['Garden_sqr_meter_per_person_in_residancy_area'] = new_test_X['Garden_sqr_meter_per_person_in_residancy_area'] /2
    new_test_X['Number_of_valued_Kneset_members'] = new_test_X['Number_of_valued_Kneset_members'] - 0.23


    pred = clf.predict(new_test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print ""
    

pick_a_new_winner(df_train_X, df_train_Y, df_test_X, df_test_Y)


# ### Plot Coaltion vs Oposition

# In[37]:


for party in coalition:
    df_party = df_train_X.dropna().copy()
    df_party['Vote'] = df_train_Y
    df_party = df_party[df_party['Vote'] == party]
    df_party[improtant_features].plot(kind='kde') 
    p.title(party)
    p.show()
    print df_party[improtant_features].mean()
    


# In[72]:



df_party = df_train_X.dropna().copy()
df_party['Vote'] = df_train_Y
df_party = df_party[df_party['Vote'].isin(coalition)]
df_party[improtant_features].plot(kind='kde') 
p.title(" Coalition")
p.show()

print df_party[improtant_features].mean()

df_party = df_train_X.dropna().copy()
df_party['Vote'] = df_train_Y
df_party = df_party[df_party['Vote'].isin(non_coalition)]
df_party[improtant_features].plot(kind='kde') 
p.title(" Non Coalition")
p.show()

print df_party[improtant_features].mean()
    


# In[73]:


def hist_plot(feat):
    for f in feat:
        df_party = df_train_X.dropna().copy()
        df_party['Vote'] = df_train_Y
        df_party = df_party[df_party['Vote'].isin(coalition)]
        df_party[f].plot(kind='hist') 
        p.title(f + " Coalition")
        p.show()

        df_party = df_train_X.dropna().copy()
        df_party['Vote'] = df_train_Y
        df_party = df_party[df_party['Vote'].isin(non_coalition)]
        df_party[f].plot(kind='hist') 
        p.title(f + " Non Coalition")
        p.show()

hist_plot(['Overall_happiness_score',
 'Is_Most_Important_Issue_Foreign_Affairs',
 'Is_Most_Important_Issue_Other',
 'Yearly_IncomeK',
 'Garden_sqr_meter_per_person_in_residancy_area',
'Will_vote_only_large_party','Avg_Satisfaction_with_previous_vote','Is_Most_Important_Issue_Military','Number_of_valued_Kneset_members'])


# In[132]:


def change_coalition(train_X, train_Y, test_X, test_Y):
    clf = RandomForestClassifier(min_samples_split=4, random_state=0)
    clf.fit(train_X, train_Y)

    pred = clf.predict(test_X)    
    distribution = Counter(pred)

    print "Original winner is party ## %s ##" % distribution.most_common(1)[0][0]
    pred = clf.predict(test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print distribution
    df_pred = pd.DataFrame()
    df_pred['Vote'] = pred
    count_coalition(test_X, df_pred['Vote'])
    print ""
    

    new_test_X = test_X.copy()
    new_test_X['Will_vote_only_large_party'] = new_test_X['Will_vote_only_large_party'] / 3 + 0.66


    pred = clf.predict(new_test_X)    
    distribution = Counter(pred)
    print "winner is party ## %s ##" % distribution.most_common(1)[0][0]
    print distribution
    df_pred = pd.DataFrame()
    df_pred['Vote'] = pred
    count_coalition(new_test_X, df_pred['Vote'])
    print ""





# Manupulating data to change coalition

# In[133]:


change_coalition(df_train_X, df_train_Y, df_test_X, df_test_Y)


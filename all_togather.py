
# coding: utf-8

# # Putting it all togather

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations

from sklearn.cross_validation import KFold

from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, scale, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier

pd.options.mode.chained_assignment = None


# In[2]:


def import_data():
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv("./data/ElectionsData-full.csv", header=0)

    df['split'] = 0
    indices = KFold(n=len(df), n_folds=5, shuffle=True)._iter_test_indices()
    df['split'][indices.next()] = 1
    df['split'][indices.next()] = 2
    raw_data = df.copy()

    raw_data[raw_data['split'] == 0].drop('split', axis=1).to_csv('./data/output/raw_train.csv', index=False, sep=',')
    raw_data[raw_data['split'] == 1].drop('split', axis=1).to_csv('./data/output/raw_test.csv', index=False, sep=',')
    raw_data[raw_data['split'] == 2].drop('split', axis=1).to_csv('./data/output/raw_validation.csv', index=False)

    return df

df_old = import_data()
df_old.info()


# # Loading the new data

# In[3]:


train_split = 0
test_split = 1
validation_split = 2
new_split = 3


# In[4]:


df_new = pd.read_csv('./data/ElectionsData_Pred_Features.csv', header=0)
df_new['split'] = new_split


# ### Looking at the data
# checking for nil values

# In[5]:


df_new.describe()


# In[6]:


df_new.info()


# __During this exersice we will try to treat the new data and the validation data the same
# Meaning we will do the same operations on both, and than measure our success with the validation data.__

# In[7]:


a = set(df_new.columns)
b = set(df_old.columns)

print "Only in df_new"
print "_" * 40
for f in sorted(a-b):
    print f

print "Only in df_validation"
print "_" * 40
for f in sorted(b-a):
    print f


# #### Looks like we have broken columns between validation and new data. lets fix that:

# In[8]:


df_new["%Of_Household_Income"] = df_new["X.Of_Household_Income"]
df_new["%Time_invested_in_work"] = df_new["X.Time_invested_in_work"]
df_new["%_satisfaction_financial_policy"] = df_new["X._satisfaction_financial_policy"]
df_new["Financial_balance_score_(0-1)"] = df_new["Financial_balance_score_.0.1."]

df_new.drop(['X.Of_Household_Income', 'X.Time_invested_in_work', 'X._satisfaction_financial_policy', 'Financial_balance_score_.0.1.'],inplace=True, axis=1)


# ## Preprocessing the new data and the Validation set
# 
# first, we want to "hide" unwanted colums from each dataset: 
# - df_validation - has the "Vote" column 
# - df_new - has the "IdentityCard_Num" column.
# 
# we will remove those for the data cleansing process, and glue them back after we're done.
# We decided to group all the data together for the data cleansing and filling data, that we will split it apart again.

# In[9]:


df_all = df_new.append(df_old, ignore_index=True)

df_Votes = df_all['Vote']
df_Ids = df_all['IdentityCard_Num']
df_split = df_all['split']

df_all = df_all.drop(['IdentityCard_Num', 'Vote','split'], axis=1)


# #### Grouping out features

# In[10]:


def group_features(_df):
    _df = _df.dropna()
    all_features = _df.columns
    
    exclude = {'Vote', 'IdentityCard_Num', 'split'}
    categorical_features = list(set(_df.select_dtypes(include=["object"]).columns) - exclude)
    numeric_features = list(set(_df.select_dtypes(exclude=["object"]).columns) - exclude)

    return [all_features, categorical_features, numeric_features]

all_features, categorical_features, numeric_features = group_features(df_all)


# #### Categorial Features

# In[11]:


for f in categorical_features:
    print f


# #### Numeric Features
# 

# In[12]:


for f in numeric_features:
    print f


# In[13]:


def rows_with_nan(_df):
    return Counter(_df.isnull().sum(axis=1).tolist())

def plot_rows_with_nan(_df, name=""):
    counter = rows_with_nan(_df)
    labels, histogram = zip(*counter.most_common())
    fig1, ax1 = plt.subplots()
    ax1.pie(histogram, labels=labels,
            colors = ['green', 'yellowgreen', 'yellow','orange', 'red'],
            autopct = lambda(p): '{:.0f}  ({:.2f}%)'.format(p * sum(histogram) / 100, p))
    ax1.axis('equal')
    plt.title(name)
    plt.show()
    


# In[14]:


plot_rows_with_nan(df_all, "Null Count")


# We can see that the new data has a lot if missing data.
# Lets try to fill it.

# # Filling the Data
# In ex2 we showed that Most_Important_Issue is a mupltiplication of Last_school_grades.
# lets fill with this info the missing values

# In[15]:


def fill_most_important_issue_and_grades_boris(df):
    translate_map = {
        'Military': 30.0,
        'Healthcare': 80.0,
        'Environment': 90.0,
        'Financial': 60.0,
        'Education': 100.0,
        'Foreign_Affairs': 40.0,
        'Social': 70.0,
        'Other': 50.0
    }

    inv_map = {v: k for k, v in translate_map.iteritems()}

    for index, row in df.iterrows():
        if pd.isnull(row['Last_school_grades']) and pd.isnull(row['Most_Important_Issue']):
            continue
        if pd.isnull(row['Last_school_grades']):
            df.loc[index, ['Last_school_grades']] = translate_map[row['Most_Important_Issue']]
        if pd.isnull(row['Most_Important_Issue']):
            df.loc[index, ['Most_Important_Issue']] = inv_map[row['Last_school_grades']]
    
    return df
    
df_all = fill_most_important_issue_and_grades_boris(df_all)

plot_rows_with_nan(df_all, "Null Count")



# ### Using Pearson's Correlation Coefficient
# 
# At this stage we are looking for corrlation between features
# We've set a threshold of 0.7 for the pearson's correlation coefficient. 
# Any two features with a value of that and above will be considered "highly correlated" and will be used to to fill the data by multiplying the value from one column by that ratio and set it to the other.
# 
# Since most of the columns are either complete or have at most 1 missing value, this will be a very useful tool.
# 
# Our method will start from the most corrlated features till it gets to 0.7.

# In[16]:



def fill_numeric_by_correlation(_df, factor, features):
    cor = _df[features].dropna().corr()
    print "We choose " + str(factor) + " correlation as correlated"
    indices = np.where(cor > factor)
    indices = [(cor.index[x], cor.columns[y], cor.loc[cor.index[x],cor.columns[y]]) for x, y in zip(*indices) if  x < y]
    indices = sorted(indices, key=lambda pair: pair[2], reverse=True)
    for pair in indices:
        fill_f1_with_f2(_df, pair[0], pair[1], pair[2])


def fill_f1_with_f2(_df, f1, f2, val):
    ratio = _df[f1].mean() / _df[f2].mean()
    print 'Filling ' + f1 + ' with ' + f2 + ' due to correlation of %f'% val
    for index, row in _df[_df[f1 + "_fill"].isnull()].iterrows():
        if ~np.isnan(_df[f2][index]):
            _df[f1 + "_fill"][index] = _df[f2][index] * ratio
    
    ratio = _df[f2].mean() / _df[f1].mean()
    print 'Filling ' + f2 + ' with ' + f1 + ' due to correlation %f'% val
    for index, row in _df[_df[f2 + "_fill"].isnull()].iterrows():
        if ~np.isnan(_df[f1][index]):
            _df[f2 + "_fill"][index] = _df[f1][index] * ratio

def remove_fill(_df, features):
    for f in features:
        _df[f] = _df[f + "_fill"]
        del _df[f + "_fill"]

def create_fill(_df, features):
    for f in features:
        _df[f + "_fill"] = _df[f]


# In[17]:


corr_factor = 0.7
print "**" * 40
create_fill(df_all, numeric_features)
fill_numeric_by_correlation(df_all, corr_factor, numeric_features)
remove_fill(df_all, numeric_features)
plot_rows_with_nan(df_all, "Null Count")


# ### Lets check the missing values status now
# looks like we made an improvment, but there is still a way to go.

# In[18]:


# Helper From Ex02

def train_on_feature(df, feature):
    train_data_X = df.drop([feature], axis=1).values
    train_data_Y = df[feature].values
    clf = RandomForestRegressor(n_estimators=50)
      
    return clf.fit(train_data_X, train_data_Y)

def build_clf(df, feature):
    df_tmp = df.copy()
    df_tmp_noNaN = df_tmp.dropna()
    print feature
    print len(df_tmp[pd.isnull(df_tmp[feature])])
    to_predict = df_tmp[pd.isnull(df_tmp[feature])].drop([feature],axis=1)
    clf = train_on_feature(df_tmp_noNaN, feature)
    return to_predict, clf


def fill_numerical(_df, numeric_features):
    results = {}
    df_numeric= _df[numeric_features]
    
    indeces_with_less_than_2_nan = [df_numeric.loc[k].isnull().sum() < 2 for k in df_numeric.index.values]
    df_for_prediction = df_numeric.loc[indeces_with_less_than_2_nan]
    
    for feature in numeric_features:
        to_predict, clf = build_clf(df_for_prediction, feature) 
        if to_predict.shape[0] > 0:           
            results[feature] = zip(to_predict.index.values, clf.predict(to_predict))
    
    #fill in the missing values
    for feature, res in results.iteritems():
        for item in res:
            _df[feature][item[0]] = item[1]
    return _df

# fill_numerical(df_all, numeric_features)
# plot_rows_with_nan(df_all, "Null Count")


# In[19]:


def fill_numeric_features_with_mean(_df, features):
    for f in features:
        _df[f + "_fill"].fillna(_df[f + "_fill"].median(), inplace=True)


# In[20]:


create_fill(df_all, numeric_features)
fill_numeric_features_with_mean(df_all, numeric_features)
remove_fill(df_all, numeric_features)
plot_rows_with_nan(df_all, "Null Count")


# __Now all there is left is the categorical data__

# In[21]:


def fill_cat_simple(df, categorical_features):
    for f in categorical_features:
        df[f].fillna(Counter(df[f].dropna()).most_common(1)[0][0], inplace=True)
        
fill_cat_simple(df_all, categorical_features)
plot_rows_with_nan(df_all, "Null Count")


# # Looks like we are done with data imputation
# ### Now we can transform and scale the data
# #### Lets look what we have

# In[22]:


df_all.info()


# In[23]:


def transform_bool(_df, name):
    _df[name] = _df[name].map({'No': -1, "Maybe": 0, 'Yes': 1}).astype(float)


def transform_category(_df, name):
    for cat in _df[name].unique():
        _df["Is_" + name + "_" + cat] = (_df[name] == cat).astype(float)
    del _df[name]


def transform_manual(_df):
    _df["Age_group"] = _df["Age_group"].map({'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(float)
    _df["Voting_Time"] = _df["Voting_Time"].map({'By_16:00': 0, 'After_16:00': 1}).astype(float)
    _df["Gender"] = _df["Gender"].map({'Male': -1, 'Female': 1}).astype(float)

    transform_bool(_df, "Looking_at_poles_results")
    transform_bool(_df, "Married")
    transform_bool(_df, "Financial_agenda_matters")
    transform_bool(_df, "Will_vote_only_large_party")
    transform_category(_df, "Most_Important_Issue")
    transform_category(_df, "Occupation")
    transform_category(_df, "Main_transportation")
    
transform_manual(df_all)


# In[24]:


def scale_numeric(_df, features):
    for f in features:
        _df[f] = (_df[f] - _df[f].min()) / (_df[f].max() - _df[f].min())
        
scale_numeric(df_all, numeric_features)


# In[25]:


df_all.describe()


# In[26]:


# df_Votes = df_all['Vote']
# df_Ids = df_all['IdentityCard_Num']
# df_split = df_all['split']

df_all['split'] = df_split
df_all['Vote'] = df_Votes
df_all['IdentityCard_Num'] = df_Ids

# train_split = 0
# test_split = 1
# validation_split = 2
# new_split = 3

df_train = df_all[df_all['split'] == train_split]
df_test = df_all[df_all['split'] == test_split]
df_validation = df_all[df_all['split'] == validation_split]

df_init_data = df_all[df_all['split'] != new_split]
df_new = df_all[df_all['split'] == new_split]


# In[27]:


df_train_X = df_train.drop(['split', 'Vote', 'IdentityCard_Num'],axis=1)
df_train_Y = df_train['Vote']

df_test_X = df_test.drop(['split', 'Vote', 'IdentityCard_Num'],axis=1)
df_test_Y = df_test['Vote']

df_validation_X = df_validation.drop(['split', 'Vote', 'IdentityCard_Num'],axis=1)
df_validation_Y = df_validation['Vote']

df_init_data_X = df_init_data.drop(['split', 'Vote', 'IdentityCard_Num'],axis=1)
df_init_data_Y = df_init_data['Vote']


df_new_Ids = df_new['IdentityCard_Num']
df_new_X = df_new.drop(['split', 'Vote', 'IdentityCard_Num'],axis=1)


# In[28]:


fixes_useful_features = [
    'Avg_Satisfaction_with_previous_vote',
    'Number_of_valued_Kneset_members',
    'Yearly_IncomeK',
    'Overall_happiness_score',
    'Avg_monthly_expense_when_under_age_21',
    'Will_vote_only_large_party',
    'Garden_sqr_meter_per_person_in_residancy_area',
    'Is_Most_Important_Issue_Other',
    'Is_Most_Important_Issue_Financial',
    'Is_Most_Important_Issue_Environment',
    'Is_Most_Important_Issue_Military',
    'Is_Most_Important_Issue_Education',
    'Is_Most_Important_Issue_Foreign_Affairs',
    'Is_Most_Important_Issue_Social'
]

df_test_X = df_test_X[fixes_useful_features]
df_train_X = df_train_X[fixes_useful_features]
df_init_data_X = df_init_data_X[fixes_useful_features] 
df_new_X = df_new_X[fixes_useful_features]


# #### We pick the same algorithm we did in Ex03, lets see if it still works

# In[29]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


clf = RandomForestClassifier(min_samples_split=4, n_estimators=100, random_state=0)
pred = cross_val_predict(clf, df_train_X.values, df_train_Y.values, cv=30)
print "***** %s *****" % clf.__class__.__name__
print classification_report(df_train_Y.values, pred, digits=5)


clf = RandomForestClassifier(min_samples_split=4, n_estimators=100, random_state=0)
pred = cross_val_predict(clf, df_init_data_X.values, df_init_data_Y.values, cv=30)
print "***** %s *****" % clf.__class__.__name__
print classification_report(df_init_data_Y.values, pred, digits=5)



# Around 96%, looks good!

# Lets try it on the test data

# In[30]:


clf = RandomForestClassifier(min_samples_split=4, n_estimators=100, random_state=0)
clf.fit(df_train_X.values, df_train_Y.values)

pred = clf.predict(df_test_X.values)
distribution = Counter(pred)
print distribution

print "predicted winner is party ## %s ##" % distribution.most_common(1)[0][0]


# In[31]:


def evaluate_miss_rate(target, pred):
    miss_vals = []
    real_vals = []
    toples = []

    miss_count = 0
    for i, j in enumerate(pred):
        if pred[i] != target[i]:
            miss_vals.append(pred[i][0])
            real_vals.append(target[i][0])
            toples.append((pred[i][0],target[i][0]))
            miss_count = miss_count + 1

    print "Total Wrong predictions %d out of %d, hit rate: %f"% (miss_count, target.size, 100 - miss_count/float(target.size) * 100) + '%'
    
evaluate_miss_rate(df_test_Y.values, pred)


# Almost 95%, looks good enough

# ### Prediction part
# We picked RandomForestClassifier with min_samples_split=4 as we did in Ex03, lets try and see who wins the elections on the new data.

# In[32]:


clf = RandomForestClassifier(min_samples_split=4, n_estimators=200, random_state=0)
clf.fit(df_init_data_X.values, df_init_data_Y.values)

PredictVote = clf.predict(df_new_X.values)
distribution = Counter(PredictVote)
print distribution

print "predicted winner is party ## %s ##" % distribution.most_common(1)[0][0]


# ## The Browns will win!!
# 
# Now lets glue up some stuff to export the prediction and try to cluster

# In[33]:


new_data = df_new_X
new_data['Vote'] = PredictVote

new_data_X = df_new_X
new_data_Y = new_data['Vote']

new_data.head(10)


# In[34]:



export_data = pd.DataFrame()
export_data['IdentityCard_Num'] = df_new['IdentityCard_Num'].astype(int)
export_data['PredictVote'] = PredictVote
export_data.to_csv('./predicted_new.csv', index=False, sep=',')


# ### Vote Distribution

# In[35]:


print "Vote distribution"
vote_dist = Counter(PredictVote)
common = vote_dist.most_common()
for party in common: 
    print "%s - Votes: %d - Percents: %0.2f%%"%(party[0], party[1], float(party[1])/ len(PredictVote) * 100.0)


# In[36]:


c = vote_dist.most_common()
parties, votes = zip(*[x for x in c])

colors = [x[:-1] for x in parties]
fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.pie([x[1] for x in c], labels=parties, colors=colors, autopct=lambda(p): '{:.0f}'.format(p * PredictVote.size / 100),
        shadow=True, radius=20)
ax1.axis('equal')
fig1.suptitle('Number of predicted votes',fontsize=20)

total_votes = new_data.shape[0]
ratios = [float(v)/total_votes for v in votes]
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.pie(ratios, labels=parties, colors=colors, autopct='%00.1f%%', shadow=True)
ax2.axis('equal')
fig2.suptitle('Percentage of predicted votes',fontsize=20)

plt.show()


# # Clustering
# This is basically what we did in ex04
# There we dicided to go for K=10 as k for KMeans

# In[37]:


k = 10


# In[38]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=k, verbose=0, random_state=0)
print "Training: K=%d" % k
km.fit(df_init_data_X)
print "Done"


# In[39]:


def k_means_party_distribution(clf, X, Y, k):
    df_Y = Y.copy().to_frame()
    df_Y['cluster'] = clf.labels_
        
    res = {}
    for i in range(0, k):
        my_df = df_Y[df_Y['cluster'] == i]
        c = Counter(my_df['Vote'].values)
        res[i] = c
    return res


# In[40]:


def basic_distribution(dist):
    for key,val in dist.iteritems():
        items = val.most_common()
        keys = []
        values = []
        for item in items:
            keys.append(item[0])
            values.append(item[0])

        print "Group: %s, Distribution: %s"%(str(key), sorted(keys))


# In[41]:


dist_per_cluster = k_means_party_distribution(km, df_init_data_X, df_init_data_Y, k)

basic_distribution(dist_per_cluster)


# Looks lide there is a small leak of parties, lets check how big is it.

# In[42]:


def k_means_cluster_distribution(clf, X, Y):
    df_Y = Y.copy().to_frame()
    df_Y['cluster'] = clf.labels_
        
    res = {}
    
    for i in Y.unique():
        my_df = df_Y[df_Y['Vote'] == i]
        c = Counter(my_df['cluster'].values)
        res[i] = c
    return res


dist_per_party = k_means_cluster_distribution(km ,df_init_data_X, df_init_data_Y)
basic_distribution(dist_per_party)


# In[43]:


print ("_" *22) + "dist per cluster" + ("_" *22)
print "_" * 60
for i in dist_per_cluster:
    print i,dist_per_cluster[i]
    
print 
print ("_" *22) + " dist per party " + ("_" *22)
print "_" * 60    
for i in dist_per_party:
    print i,dist_per_party[i]


# Looks like the Red leaked a little bit to cluster 0 (2 votes)
# 
# And the yellows leaked to clusters (4, 0 and 2)
# 
# But it is only in very small number, maybe it is because we did the outlier detection little bit differently
# 
# we will ignore the leak in this part

# In[44]:


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


# In[45]:


from operator import itemgetter

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


# In[46]:


coalition = ['Purples', 'Browns', 'Greens', 'Pinks', 'Whites']
non_coalition = ['Greys', 'Oranges', 'Reds', 'Yellows', 'Blues']
coalition_clusters = [0,4,5,2,3]


# In[47]:


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

    
count_coalition(df_init_data_X, df_init_data_Y) 


# In[48]:


count_coalition(df_new_X.drop(['Vote'], axis=1), df_new_X['Vote'])


# In[49]:



def predict_clustering(clf, X):
    cluster_pred = clf.predict(X)
    return cluster_pred

def count_coalition_clusters(pred):
    c = Counter(pred)
    in_coal = sum(c[cid] for cid in coalition_clusters)
    total = sum(c.values())  
    print "Clustering - In Coualtion clusters the are %d votes which are %.02f%% percent"%(in_coal, float(in_coal) / total * 100)

clustering_prediction = predict_clustering(km, df_new_X.drop(['Vote'], axis=1))  
count_coalition_clusters(clustering_prediction)


# In[50]:


def in_coal_and_coal_cluster(row):
    if row["Vote"] in coalition and row["cluster"] in coalition_clusters:
        return 1
    else:
        return 0
    
    
def count_coalition_with_cluster(_df, pred):
    df = _df.copy()
    df['cluster'] = pred

    df['coal+cluster'] = df.apply (lambda row: in_coal_and_coal_cluster(row),axis=1)
    val_counts = df['coal+cluster'].value_counts()
    print "In Coualtion + coalition clusters the are %d votes which are %.02f%% percent"%(val_counts[1], float(val_counts[1]) / len(_df) * 100)

clustering_prediction = predict_clustering(km, df_new_X.drop(['Vote'], axis=1))  
count_coalition_with_cluster(df_new_X, clustering_prediction)
    


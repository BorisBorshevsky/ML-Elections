
# coding: utf-8

# In[72]:


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



# In[113]:


def load_prepared_data():
	df_train = pd.read_csv('./data/output/processed_train.csv', header=0)
	df_test = pd.read_csv('./data/output/processed_test.csv', header=0)
	features = list(set(df_train.columns) - {'Vote'})
	target = 'Vote'

	df_train_X = df_train[features]
	df_train_Y = df_train[target]
	df_test_X = df_test[features]
	df_test_Y = df_test[target]
# 	labels = {"0":"Blues","1":"Browns","2":"Greens","3":"Greys","4":"Oranges","5":"Pinks","6":"Purples","7":"Reds","8":"Whites","9":"Yellows" }
 	labels = ["Blues","Browns","Greens","Greys","Oranges","Pinks","Purples","Reds","Whites","Yellows"]
	return df_train_X, df_train_Y, df_test_X, df_test_Y, labels


# In[114]:


df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()

train_val_data = pd.concat([df_train_X])
features = train_val_data.values
target = pd.concat([df_train_Y]).values


# In[115]:


clf = SVC(kernel='linear')
scores = cross_val_score(clf, features, target, cv=15)
print "linear %s score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[116]:


clf = LinearSVC(multi_class='ovr')
scores = cross_val_score(clf, features, target, cv=15)
print "%s OVR Score: %f, std: %f" % (clf.__class__.__name__,np.mean(scores), np.std(scores))


# In[117]:


clf = LinearSVC(multi_class='crammer_singer')
scores = cross_val_score(clf, features, target, cv=15)
print "%s crammer_singer Score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[144]:



clf = OneVsOneClassifier(LinearSVC())
scores = cross_val_score(clf, features, target, cv=15)
print "%s Score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[79]:


clf = GaussianNB()
scores = cross_val_score(clf, features, target, cv=15)
print "%s Score: %f" % (clf.__class__.__name__, np.mean(scores))


# In[146]:


all_scores = []
for splitter in range(2,20):
    clf = DecisionTreeClassifier(min_samples_split=splitter, random_state=0)
    scores = cross_val_score(clf, features, target, cv=15)
    score = np.mean(scores)
    all_scores.append(score)
    print "minimum splitter = %d, score = %f" % (splitter, score)
print "Best Splitter size: %d" % (np.argmax(all_scores) + 2)
print "%s with best splitter: %f" % (clf.__class__.__name__, all_scores[np.argmax(all_scores)])

clf = DecisionTreeClassifier(random_state=0)
scores = cross_val_score(clf, features, target, cv=15)
score = np.mean(scores)
print "%s Default score: %f"% (clf.__class__.__name__, score)


# In[120]:


all_scores = []
for splitter in range(2,20):
    clf = RandomForestClassifier(min_samples_split=splitter, random_state=0)
    scores = cross_val_score(clf, features, target, cv=15)
    score = np.mean(scores)
    all_scores.append(score)
    print "minimum splitter = %d, score = %f" % (splitter, score)
print "Best Splitter size: %d" % (np.argmax(all_scores) + 2)
print "%s with best splitter: %f" % (clf.__class__.__name__, all_scores[np.argmax(all_scores)])

clf = RandomForestClassifier(random_state=0)
scores = cross_val_score(clf, features, target, cv=15)
score = np.mean(scores)
print "%s Default score: %f"% (clf.__class__.__name__, score)


# In[145]:


all_scores = []
for n in range(2,20):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, features, target, cv=15)
    score = np.mean(scores)
    all_scores.append(score)
    print "minimum k_neighbors = %d, score = %f" % (n, score)
print "Best n_neighbors size: %d" % (np.argmax(all_scores) + 2)
print "KNeighborsClassifier with best N param: %f" % (all_scores[np.argmax(all_scores)])


# In[122]:


clf = Perceptron(max_iter=300)
scores = cross_val_score(clf, features, target, cv=10)
print "%s Score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[123]:


clf = LinearDiscriminantAnalysis()
scores = cross_val_score(clf, features, target, cv=10)
print "%s Score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[124]:


clf = RandomForestClassifier(random_state=0)
scores = cross_val_score(clf, features, target, cv=10)
print "%s Score: %f, std: %f" % (clf.__class__.__name__, np.mean(scores), np.std(scores))


# In[87]:


clf = MLPClassifier(verbose=0, activation='relu', hidden_layer_sizes=(50, 25, 10), 
                  random_state=0, max_iter=500, solver='sgd', 
                  learning_rate='invscaling', momentum=.9,
                  nesterovs_momentum=True, learning_rate_init=0.2)
scores = cross_val_score(clf, features, target, cv=10)
print "MLPClassifier Score: %f, std: %f" % (np.mean(scores), np.std(scores))


# In[147]:


clf = DecisionTreeClassifier(min_samples_split=8, random_state=0)
pred = cross_val_predict(clf, features, target, cv=30)
print "***** %s *****" % clf.__class__.__name__
print classification_report(target, pred, target_names=labels, digits=5)


# In[148]:


clf = KNeighborsClassifier(n_neighbors=3)
pred = cross_val_predict(clf, features, target, cv=30)
print "***** %s *****" % clf.__class__.__name__
print classification_report(target, pred, target_names=labels, digits=5)


# In[150]:


clf = RandomForestClassifier(min_samples_split=4, random_state=0)
pred = cross_val_predict(clf, features, target, cv=30)
print "***** %s *****" % clf.__class__.__name__
print classification_report(target, pred, target_names=labels, digits=5)


# In[186]:


print "Estimating DecisionTreeClassifier"
k_fold = RepeatedStratifiedKFold(n_splits=10)
clf_tree = DecisionTreeClassifier(min_samples_split=8)
a = []
for train_indices, test_indices in k_fold.split(features, target):
    clf_tree.fit(features[train_indices], target[train_indices])
    a.append(clf_tree.score(features[test_indices],target[test_indices]))
    
print "training score, mean: %f"% (np.array(a).mean())    


# In[93]:


print "Estimating KNeighborsClassifier"
k_fold = RepeatedStratifiedKFold(n_splits=10)
clf_knn = KNeighborsClassifier(n_neighbors=5)
a = []
for train_indices, test_indices in k_fold.split(features, target):
    clf_knn.fit(features[train_indices], target[train_indices])
    a.append(clf_knn.score(features[test_indices],target[test_indices]))
    
print "training score, mean: %f"% (np.array(a).mean())    


# In[94]:


print "Estimating RandomForestClassifier"
k_fold = RepeatedStratifiedKFold(n_splits=5, random_state=0)
clf_random_forest = RandomForestClassifier(min_samples_split=4, max_features=None, random_state=0)
a = []
for train_indices, test_indices in k_fold.split(features, target):
    clf_random_forest.fit(features[train_indices], target[train_indices])
    a.append(clf_random_forest.score(features[test_indices],target[test_indices]))
    
print "training score, mean: %f"% (np.array(a).mean())    


# In[95]:


features_test = df_test_X
target_test = df_test_Y


# In[204]:


# clf = RandomForestClassifier(min_samples_split=4, random_state=0)
# clf.fit(features, target)

# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(features, target)

clf = DecisionTreeClassifier(min_samples_split=8)
clf.fit(features, target)




# In[205]:



pred = clf.predict(features_test)

distribution = np.bincount(pred.astype('int64'))
most_common = np.argmax(distribution)

print "winner is party ## %s ##" % labels[most_common.astype('int')]


# In[206]:


print "Vote distribution"
distribution = np.bincount(pred.astype('int64'))

for index,party in enumerate(distribution):
    print "%s, %f, %f"%(labels[index], distribution[index], distribution[index]/ float(target_test.size) * 100) + '%'


# In[207]:


target_test_labled = target_test.map(lambda x: labels[int(x)])
pred_test_labled = pd.DataFrame(pred).applymap(lambda x: labels[int(x)])

print(classification_report(target_test_labled, pred_test_labled, target_names=labels))


# In[209]:


print labels
confusion_matrix(target_test_labled, pred_test_labled, labels=labels)


# In[210]:


pred1 = pred_test_labled.values   
target1 = pd.DataFrame(target_test_labled).values

miss_vals = []
real_vals = []
toples = []

miss_count = 0
for i, j in enumerate(pred1):
    if pred1[i] != target1[i]:
        miss_vals.append(pred1[i][0])
        real_vals.append(target1[i][0])
        toples.append((pred1[i][0],target1[i][0]))
        miss_count = miss_count + 1


print "Total Wrong predictions %d out of %d, hit rate: %f"% (miss_count, target1.size, 100 - miss_count/float(target1.size) * 100) + '%'


# In[158]:


pred_test_labled.to_csv("./data/output/test_predictions.csv",header=['Vote'] ,index=False)


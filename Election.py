
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

#set your working_dir
working_dir = "C:/Users/sfine01/Documents/Python Scripts/Exercises/Titanic"

# For .read_csv, always use header=0 when you know row 0 is the header row
train = pd.read_csv(working_dir+"/data/train.csv", header=0)
test = pd.read_csv(working_dir+"/data/test.csv", header=0)


# ## First look at the data

# In[2]:

train.head()


# ## VARIABLE DESCRIPTIONS:
# * survival - Survival (0 = No; 1 = Yes)
# * pclass   - Passenger Class  (1 = 1st; 2 = 2nd; 3 = 3rd)
# * name     - Name
# * sex      - Sex
# * age      - Age
# * sibsp    -       Number of Siblings/Spouses Aboard
# * parch    -       Number of Parents/Children Aboard
# * ticket   -       Ticket Number
# * fare     -       Passenger Fare
# * cabin    -       Cabin
# * embarked -       Port of Embarkation  (C = Cherbourg; Q = Queenstown; S = Southampton)
#                 
# 
# ### SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
#  If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# * Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# * Spouse :   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# * Parent :   Mother or Father of Passenger Aboard Titanic
# * Child  :    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.

# In[3]:

#Training set feature types
train.dtypes


# In[4]:

#Concise summary of training data
train.info()


# In[5]:

#Summary statistics of training data, excluding NaN values
train.describe()


# caution: there are a lot of missing values in Age, for example (see info()) 
# Since the nulls were left out of the calculation, the summary statistics values 
# e.g. the mean) might be miskeading
# 

# In[6]:

#Any files you save will be available in the output tab below
train.to_csv(working_dir+"/data/output/copy_of_the_training_data.csv", index=False)


# # Simple data exploration

# In[7]:

#number of distinct Pclass values
PclassNum = train.Pclass.unique().size

#The count of males in each class
for i in range(1,PclassNum+1):
    print i, len(train[ (train['Sex'] == 'male') & (train['Pclass'] == i) ])


# In[8]:

#let's look at the histogram of ages calculated without missing values
import pylab as P
train.Age.dropna().hist(bins=16, range= (0,80), alpha=0.5)
P.show()


# ## Preparing the data for Machine Learning
# ### Cleaning the data, creating new features, transforming to numeric values, droping NaNs

# #### "Sex" to int (binary)
# > Transform to **Int** and store in a new 'Gender' columns

# In[9]:

#Adding in new 'Gender' column to the dataframe
train['Gender'] = train['Sex'].map( {'female':0, 'male':1}).astype(int)
test['Gender'] = test['Sex'].map( {'female':0, 'male':1}).astype(int)

#Let's creat a crosstabcross-tabulation to look at this transformation
pd.crosstab(train.Gender, train.Sex, rownames=['Gender'], colnames=['Sex'])


# #### "Embarked" is multicategorical - Transforme in two steps
# 1. Transform to **categoricl** and store in new 'Embarkport' columns
# 2. Transfom to **Int** and store in new 'EmbarkportInt' columns

# In[10]:

train['Embarkport'] = train['Embarked'].astype("category")
train['EmbarkportInt'] = train['Embarkport'].cat.rename_categories(range(train['Embarkport'].nunique())).astype(int)
test['Embarkport'] = test['Embarked'].astype("category")
test['EmbarkportInt'] = test['Embarkport'].cat.rename_categories(range(test['Embarkport'].nunique())).astype(int)

#Let's creat a crosstabcross-tabulation to look at this transformation
pd.crosstab(train.EmbarkportInt, train.Embarked, rownames=['EmbarkportInt'], colnames=['Embarked'])


# In[11]:

#Cleaning - drop the newly created columns
train = train.drop(['Embarkport','EmbarkportInt'], axis=1) 
test = test.drop(['Embarkport','EmbarkportInt'], axis=1) 


# ## convert non-numeric data

# In[12]:

# Identify which of the orginal features are objects
ObjFeat=train.keys()[train.dtypes.map(lambda x: x=='object')]

# Transform the original features to categorical
# Creat new 'int' features, resp.
for f in ObjFeat:
    train[f] = train[f].astype("category")
    train[f+"Int"] = train[f].cat.rename_categories(range(train[f].nunique())).astype(int)
    train.loc[train[f].isnull(), f+"Int"] = np.nan #fix NaN conversion

    # Let's creat a crosstabcross-tabulation to look at this transformation
    # pd.crosstab(train[f+"Int"], train[f], rownames=[f+"Int"], colnames=[f])
        
    test[f] = test[f].astype("category")        
    if test[f].cat.categories.isin(train[f].cat.categories).all():
        test[f] = test[f].cat.rename_categories(train[f].cat.categories)
    else:    
        print "\n\nTrain and Test doesnt share the same set of categories in feture '", f, "'" 
    test[f+"Int"] = test[f].cat.rename_categories(range(test[f].nunique())).astype(int)
    test.loc[test[f].isnull(), f+"Int"] = np.nan #fix NaN conversion


# ## Missing Values
# > Age has missing values 
# > Create new 'AgeFill' column, where missing value is filled with the median of the gender and Pclass

# In[13]:

median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & 
                              (train['Pclass'] == j+1)]['Age'].dropna().median() 

median_ages


# In[14]:

train['AgeFill'] = train['Age']
test['AgeFill'] = test['Age']

# Fill the missin values with medians
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
        test.loc[ (test.Age.isnull()) & (test.Gender == i) & (test.Pclass == j+1),                 'AgeFill'] = median_ages[i,j]

train[ train['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)


# In[15]:

# Create a feature that records whether the Age was originally missing
train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
test['AgeIsNull'] = pd.isnull(test.Age).astype(int)

train[['Gender','Pclass','Age','AgeFill','AgeIsNull']].head(10)


# ## Feature Construction
# Parch is the number of parents or children onboard, 
# and SibSp is the number of siblings or spouses. 
# > We can aggregate them together to form a new 'FamilySize' 

# In[16]:

train['FamilySize'] = train.SibSp + train.Parch
test['FamilySize'] = test.SibSp + test.Parch


# Pclass had a large effect on survival, and it's possible Age will too.
# >A Constructed feature is the Age and Pclass multiplication, thus amplifing 
# old age and 3rd class - both were less likely to survive

# In[17]:

train['Age*Class'] = train.AgeFill * train.Pclass
test['Age*Class'] = test.AgeFill * test.Pclass

train['Age*Class'].hist()


# ## Final Preperations

# In[18]:

# categorical columns
train.dtypes[ObjFeat]
test.dtypes[ObjFeat]


# #### Drop the columns that will not be used in training

# In[19]:

train = train.drop(ObjFeat, axis=1) 
test = test.drop(ObjFeat, axis=1) 

train.dtypes


# #### Create training and testing sets that has no null (by dropping rםws ) 

# In[21]:

train_noNaN = train.dropna()
test_noNaN = test.dropna()

train_noNaN.info()


# In[22]:

test_noNaN.info()


# #### Convert to Numpy array

# In[23]:

train_data_X = train.drop(['Survived'], axis=1).values
train_data_Y = train.Survived.values
test_data = test.values

train_data_X_noNaN = train_noNaN.drop(['Survived'], axis=1).values
train_data_Y_noNaN = train_noNaN.Survived.values
test_data_noNaN = test_noNaN.values


# # Scikit-learn

# ## Feature Selection

# In[28]:

# Import feature selection package
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif

# Save original feature names
feat_names = train.drop(['Survived'], axis=1).columns.values


# ### Removing features with low variance

# In[29]:

# Remove features that are of a low variance
varsel = VarianceThreshold(threshold=0.01)
train_data_X_noNaN = varsel.fit_transform(train_data_X_noNaN)
featsel_idx = varsel.get_support()

print 'Removing features with low variance - ','\t',feat_names[~featsel_idx]


# ### Univariate feature selection

# In[43]:

# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 15% most significant features
selector = SelectPercentile(f_classif, percentile=15)
selector.fit(train_data_X_noNaN, train_data_Y_noNaN)
f_scores = selector.scores_
f_scores /= f_scores.max()

X_indices = np.arange(train_data_X_noNaN.shape[-1])

plt.bar(X_indices - .65, f_scores, width=.3,
        label='f_classif', color='g')

# Univariate feature selection with mutual information for feature scoring
selector = SelectPercentile(mutual_info_classif, percentile=15)
selector.fit(train_data_X_noNaN, train_data_Y_noNaN)
MI_scores = selector.scores_
MI_scores /= MI_scores.max()

plt.bar(X_indices - .35, MI_scores, width=.3,
        label='MI', color='b')

plt.title("Comparing Univariate Feature Selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()


# ### Wrapper Methods

# In[46]:

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# The "accuracy" scoring is proportional to the number of correct
# classifications
svc = SVC(kernel="linear", C=1)
rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
rfecv.fit(train_data_X_noNaN, train_data_Y_noNaN)

minmax = MinMaxScaler()
scores = map(float, rfecv.ranking_)
scores = minmax.fit_transform(-1.0*np.array([scores]).T).T[0]

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# ### Embedded Methods

# In[44]:

#Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf = clf.fit(train_data_X_noNaN, train_data_Y_noNaN)
tree_weights = clf.feature_importances_  
tree_weights /= tree_weights.max()

plt.bar(X_indices - .05, tree_weights, width=.3, 
        label='Tree weight', color='r')
plt.bar(X_indices - .35, MI_scores, width=.3,
        label='MI', color='b')
plt.bar(X_indices - .65, f_scores, width=.3,
        label='f_classif', color='g')


plt.title("Comparing Univariate and Embedded Feature Selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()


# In[ ]:




# ## Training Random Forest Classifier

# In[48]:

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Prepare train and test data using cross validation
X_train_noNaN, X_test_noNaN, y_train_noNaN, y_test_noNaN = train_test_split(train_data_X_noNaN, 
                                                                            train_data_Y_noNaN) 
                                                                            

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 3)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data_X_noNaN,train_data_Y_noNaN)

# output = forest.predict(test_data_noNaN)
y_pred_noNaN = forest.predict(X_test_noNaN)

print "accuracy:", metrics.accuracy_score(y_test_noNaN, y_pred_noNaN)
print "precision:", metrics.precision_score(y_test_noNaN, y_pred_noNaN)
print "recall:", metrics.recall_score(y_test_noNaN, y_pred_noNaN)
print "f1 score:", metrics.f1_score(y_test_noNaN, y_pred_noNaN)


# ## SVM

# In[25]:

from sklearn.svm import SVC

clf = SVC().fit(X_train_noNaN, y_train_noNaN)
y_pred_noNaN = clf.predict(X_test_noNaN)

print "accuracy:", metrics.accuracy_score(y_test_noNaN, y_pred_noNaN)
print "precision:", metrics.precision_score(y_test_noNaN, y_pred_noNaN)
print "recall:", metrics.recall_score(y_test_noNaN, y_pred_noNaN)
print "f1 score:", metrics.f1_score(y_test_noNaN, y_pred_noNaN)


# ### Create a K-Folds cross validation iterator

# In[26]:

from sklearn.cross_validation import KFold

n_folds=5
kf = KFold(n=train_data_X_noNaN.shape[0], n_folds=n_folds)

print(kf)  


# ### Train/Test Random Forest and SVM usin K-Fold cross validation

# In[27]:

for k, (train_index, test_index) in enumerate(kf):
    # Random Forest
    forest = forest.fit(train_data_X_noNaN[train_index], train_data_Y_noNaN[train_index])
    y_pred_noNaN_RF = forest.predict(train_data_X_noNaN[test_index])
    #SVM
    clf = SVC().fit(train_data_X_noNaN[train_index], train_data_Y_noNaN[train_index])
    y_pred_noNaN_SVM = clf.predict(train_data_X_noNaN[test_index])
    
    #results    
    print("[fold {0}] RF score: {1:.5}, SVM score: {2:.5}".
          format(k, metrics.accuracy_score(train_data_Y_noNaN[test_index],y_pred_noNaN_RF), 
                 metrics.accuracy_score(train_data_Y_noNaN[test_index], y_pred_noNaN_SVM)))
                                           


# In[ ]:




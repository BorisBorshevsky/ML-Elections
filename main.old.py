import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as p

# Import feature selection package
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif

# For Wrapper Method
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


def load_data_frame():
    working_dir = "."
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv(working_dir + "/data/ElectionsData-full.csv", header=0)

    return df

def fill_missing_values(df):
    numeric_fields = df.select_dtypes(exclude=["object"])
    categorial_fields = df.select_dtypes(include=["object"])

    # Fill numeric fields
    for f in numeric_fields:
        df[f].fillna(df[f].median(), inplace=True)

    # Fill categorial fields
    for f in categorial_fields:
        df[f].fillna(df[f].value_counts().idxmax(), inplace=True)

    return df

def transform_categorical_to_int(df):
    # Change all categorical to numeric
    objFeat = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    # Transform the original features to categorical
    # Create new 'int' features, resp.
    for f in objFeat:
        df[f] = df[f].astype("category")
        df[f + "Int"] = df[f].cat.rename_categories(range(df[f].nunique())).astype(int)
        df.loc[df[f].isnull(), f + "Int"] = np.nan    # fix NaN conversion

    return df

def filter_outliar_detection(df):
    numeric_fields = df.select_dtypes(exclude=["category"])

    threshold = 3
    for f in numeric_fields:
        std = df[f].std()
        mean = df[f].mean()
        df = df[df[f].between(mean - threshold * std, mean + threshold * std)]

    return df


def scale_numeric(df):
    numeric_fields = df.select_dtypes(exclude=["category"])

    for f in numeric_fields:
        if f != "VoteInt":
            df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())

    return df

def transform_categorical(df):
    df["Age_groupInt"] = df["Age_group"].map({'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(int)
    df["Voting_TimeInt"] = df["Voting_Time"].map({'By_16:00': 0, 'After_16:00': 1}).astype(int)
    df["GenderInt"] = df["Gender"].map({'Male': -1, 'Female': 1}).astype(int)

    def transform_bool(name):
        df[name + "_Int"] = df[name].map({'No': -1, "Maybe": 0, 'Yes': 1}).astype(int)

    def transform_category(name):
        for cat in df[name].unique():
            df["Is_" + name + "_" + cat] = (df[name] == cat).astype(int)

    transform_bool("Looking_at_poles_results")
    transform_bool("Married")
    transform_bool("Financial_agenda_matters")
    transform_bool("Will_vote_only_large_party")
    transform_category("Most_Important_Issue")
    transform_category("Occupation")
    transform_category("Main_transportation")

    return df


def to_np_array(df):
    category_features = df.select_dtypes(include=["category"])

    df_no_cat = df.drop(category_features, axis=1)

    feat_names = df_no_cat.drop(['VoteInt'], axis=1).columns.values

    # Convert to np array
    df_data_x = df_no_cat.drop(['VoteInt'], axis=1).values
    df_data_y = df_no_cat.VoteInt.values

    return df_data_x, df_data_y, feat_names


def filter_by_variance(data_x, feat_names):
    print
    varsel = VarianceThreshold(threshold=0.01)
    data_x = varsel.fit_transform(data_x)
    featsel_idx = varsel.get_support()

    print 'Removing features with low variance - ', '\t', feat_names[~featsel_idx]

    return data_x

def main():
    # set your working_dir
    df = load_data_frame()
    df = fill_missing_values(df)
    df = transform_categorical_to_int(df)
    df = filter_outliar_detection(df)
    df = scale_numeric(df)
    df = transform_categorical(df)

    df_data_x, df_data_y, feature_names = to_np_array(df)

    df_data_x = filter_by_variance(df_data_x, feature_names)
    wrapper_method(df_data_x, df_data_y, feature_names)
    univariate_feature_selection(df_data_x, df_data_y, feature_names)


def univariate_feature_selection(df_data_x, df_data_y, feature_names):
    selector = SelectPercentile(f_classif, percentile=15)
    selector.fit(df_data_x, df_data_y)
    f_scores = selector.scores_
    f_scores /= f_scores.max()

    print "F CLASSIF:"
    for idx, val in enumerate(selector.get_support()):
        if val:
            print feature_names[idx]

    selector = SelectPercentile(mutual_info_classif, percentile=15)
    selector.fit(df_data_x, df_data_y)

    print "MI:"
    for idx, val in enumerate(selector.get_support()):
        if val:
            print feature_names[idx]


def wrapper_method(df_data_x, df_data_y, feature_names):
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    svc = SVC(kernel="linear", C=1)
    rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
    rfecv.fit(df_data_x, df_data_y)

    print("Optimal features by linear SVC : %d" % rfecv.n_features_)
    for idx, val in enumerate(rfecv.get_support()):
        if val:
            print feature_names[idx]


if __name__ == '__main__':
        main()
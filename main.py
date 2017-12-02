import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as p
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif


def main():
	redundant_features = []
	useful_features = []

	df = import_data()

	all_features, categorical_features, numeric_features = group_features(df)

	fill_numeric_features(df, numeric_features)
	fill_categorical_features(df, categorical_features)
	transform_categorical_features(df, categorical_features)
	transform_label(df, "Vote")

	df = df.dropna()

	# # Change all categorial to numeric
	# ObjFeat = df.keys()[df.dtypes.map(lambda x: x == 'object')]
	#
	# # Transform the original features to categorical
	# # Creat new 'int' features, resp.
	# for f in ObjFeat:
	# 	df[f] = df[f].astype("category")
	# 	df[f + "Int"] = df[f].cat.rename_categories(range(df[f].nunique())).astype(int)
	# 	df.loc[df[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

	# Outliar detection
	threshold = 3
	for f in numeric_features:
		std = df[f].std()
		mean = df[f].mean()
		df = df[df[f].between(mean - threshold * std, mean + threshold * std)]

	# numeric_features = df.select_dtypes(exclude=["category"])

	for f in numeric_features:
		if f != "Vote":
			df[f] = (df[f] - df[f].min()) / (df[f].max() - df[f].min())

	df["Age_group_Int"] = df["Age_group"].map({'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(int)
	df["Voting_Time_Int"] = df["Voting_Time"].map({'By_16:00': 0, 'After_16:00': 1}).astype(int)
	df["Gender_Int"] = df["Gender"].map({'Male': -1, 'Female': 1}).astype(int)

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

	category_features = df.select_dtypes(include=["category"])

	df = df.drop(category_features, axis=1)

	feat_names = df.drop(['Vote', 'split'], axis=1).columns.values

	# Convert to np array
	df_data_X = df.drop(['Vote', 'split'], axis=1).values
	df_data_Y = df.Vote.values

	# Remove feature with low varriance
	varsel = VarianceThreshold(threshold=0.02)
	df_data_X = varsel.fit_transform(df_data_X)
	featsel_idx = varsel.get_support()

	print 'Removing features with low variance - ', '\t', feat_names[~featsel_idx]


	# # The "accuracy" scoring is proportional to the number of correct
	# # classifications
	# svc = SVC(kernel="linear", C=1)
	# rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
	# rfecv.fit(df_data_X, df_data_Y)
	#
	# print "RFE #1"
	# for idx, val in enumerate(rfecv.get_support()):
	# 	if val:
	# 		print feat_names[idx]
	#
	# print("Optimal number of features : %d" % rfecv.n_features_)
	#
	# # Univariate feature selection with F-test for feature scoring
	# # We use the default selection function: the 15% most significant features
	# selector = SelectPercentile(f_classif, percentile=15)
	# selector.fit(df_data_X, df_data_Y)
	# f_scores = selector.scores_
	# f_scores /= f_scores.max()
	#
	# print "F CLASSIF"
	# for idx, val in enumerate(selector.get_support()):
	# 	if val:
	# 		print feat_names[idx]
	#
	# # Univariate feature selection with mutual information for feature scoring
	# selector = SelectPercentile(mutual_info_classif, percentile=15)
	# selector.fit(df_data_X, df_data_Y)
	#
	# print "MI"
	# for idx, val in enumerate(selector.get_support()):
	# 	if val:
	# 		print feat_names[idx]
	#
	# print redundant_features

	export_transformed_data(df)


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


def export_transformed_data(_df):
	_df[_df['split'] == 0].drop('split', axis=1).to_csv('./data/output/processed_train.csv', index=False)
	_df[_df['split'] == 1].drop('split', axis=1).to_csv('./data/output/processed_test.csv', index=False)
	_df[_df['split'] == 2].drop('split', axis=1).to_csv('./data/output/processed_validation.csv', index=False)


def group_features(_df):
	_df = _df.dropna()
	all_features = _df.drop(['Vote', 'split'], axis=1).columns
	categorical_features = _df.drop(['Vote', 'split'], axis=1).select_dtypes(include=["object"])
	numeric_features = _df.drop(['Vote', 'split'], axis=1).select_dtypes(exclude=["object"])

	return [all_features, categorical_features, numeric_features]


def fill_numeric_features(_df, features):
	for f in features:
		_df[f].fillna(_df[f].median(), inplace=True)


def fill_categorical_features(_df, features):
	for f in features:
		_df[f].fillna(_df[f].value_counts().idxmax(), inplace=True)


def transform_categorical_features(_df, features):
	for f in features:
		_df[f] = _df[f].astype("category")
		_df[f + "_Int"] = _df[f].cat.rename_categories(range(_df[f].nunique())).astype(int)
		_df.loc[_df[f].isnull(), f + "+Int"] = np.nan  # fix NaN conversion3


def transform_label(_df, label):
	_df[label] = _df[label].astype("category").cat.rename_categories(range(_df[label].nunique())).astype(int)


if __name__ == '__main__':
	main()

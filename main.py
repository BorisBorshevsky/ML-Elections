from IPython import embed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as p
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

# Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif

pd.options.mode.chained_assignment = None


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


def group_features(_df, _dic):
	_df = _df.dropna()
	all_features = _df.drop(['Vote', 'split'], axis=1).columns
	categorical_features = _df.drop(['Vote', 'split'], axis=1).select_dtypes(include=["object"])
	numeric_features = _df.drop(['Vote', 'split'], axis=1).select_dtypes(exclude=["object"])

	for column in _df.columns:
		_dic[column] = column

	return [all_features, categorical_features, numeric_features]


def fill_numeric_features(_df, features):
	for f in features:
		_df[f].fillna(_df[f].median(), inplace=True)


def fill_numeric_by_correlation(_df, factor, features):
	redundant_features = []
	cor = _df[features.columns].dropna().corr()
	print "We choose " + str(factor) + " correlation as correlated"
	indices = np.where(cor > factor)
	indices = [(cor.index[x], cor.columns[y]) for x, y in zip(*indices) if x != y and x < y]
	for pair in indices:
		fill_f1_with_f2(_df, pair[0], pair[1])
		redundant_features.append(pair[1])
	return redundant_features


def fill_f1_with_f2(_df, f1, f2):
	ratio = _df[f1].mean() / _df[f2].mean()
	print 'Filling ' + f1 + ' with ' + f2 + ' due to correlation'
	for index, row in _df[_df[f1].isnull()].iterrows():
		if ~np.isnan(_df[f2][index]):
			_df[f1][index] = _df[f2][index] * ratio

	ratio = _df[f2].mean() / _df[f1].mean()
	print 'Filling ' + f2 + ' with ' + f1 + ' due to correlation'
	for index, row in _df[_df[f2].isnull()].iterrows():
		if ~np.isnan(_df[f1][index]):
			_df[f2][index] = _df[f1][index] * ratio



def fill_categorical_features(_df, features):
	for f in features:
		for index, row in _df[_df[f].isnull()].iterrows():
			most_common = _df[(_df.Vote == row['Vote'])][f].value_counts().idxmax()
			_df.at[index, f] = most_common


def transform_categorical_features(_df, _dic, features):
	for f in features:
		_df[f] = _df[f].astype("category")
		_df[f + "_Int"] = _df[f].cat.rename_categories(range(_df[f].nunique())).astype(int)
		_df.loc[_df[f].isnull(), f + "_Int"] = np.nan  # fix NaN conversion3
		_dic[f + "_Int"] = f


def transform_label(_df, label):
	_df[label] = _df[label].astype("category").cat.rename_categories(range(_df[label].nunique())).astype(int)


def outliar_detection(_df, features):
	std_threshold = 3

	for f in features:
		std = _df[f].std()
		mean = _df[f].mean()
		_df[~_df[f].between(mean - std_threshold * std, mean + std_threshold * std)] = np.nan
	return _df


def scale_numeric(_df, features):
	for f in features:
		_df[f] = (_df[f] - _df[f].min()) / (_df[f].max() - _df[f].min())


def transform_bool(_df, name):
	_df[name] = _df[name].map({'No': -1, "Maybe": 0, 'Yes': 1}).astype(int)


def transform_category(_df, _dic, name):
	for cat in _df[name].unique():
		_df["Is_" + name + "_" + cat] = (_df[name] == cat).astype(int)
		_dic["Is_" + name + "_" + cat] = name
	del _df[name]


def transform_manual(_df, _dic):
	_df["Age_group"] = _df["Age_group"].map({'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(int)
	_df["Voting_Time"] = _df["Voting_Time"].map({'By_16:00': 0, 'After_16:00': 1}).astype(int)
	_df["Gender"] = _df["Gender"].map({'Male': -1, 'Female': 1}).astype(int)

	transform_bool(_df, "Looking_at_poles_results")
	transform_bool(_df, "Married")
	transform_bool(_df, "Financial_agenda_matters")
	transform_bool(_df, "Will_vote_only_large_party")
	transform_category(_df, _dic, "Most_Important_Issue")
	transform_category(_df, _dic, "Occupation")
	transform_category(_df, _dic, "Main_transportation")


def to_np_array(_df):
	df_data_X = _df.drop(['split', 'Vote'], axis=1).values
	df_data_Y = _df.Vote.values
	features_list = _df.drop(['split', 'Vote'], axis=1).columns
	return [df_data_X, df_data_Y, features_list]


def variance_filter(data_X, features_list):
	varsel = VarianceThreshold(threshold=0.01)
	varsel.fit_transform(data_X)
	featsel_idx = varsel.get_support()
	print 'Removing features with low variance - ', '\t', list(features_list[~featsel_idx])
	return list(features_list[~featsel_idx])


def select_features_with_rfe(data_X, data_Y, feature_names):
	result = []

	svc = SVC(kernel="linear", C=1)
	rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
	rfecv.fit(data_X, data_Y)

	print("RFE - Optimal number of features : %d" % rfecv.n_features_)

	for idx, val in enumerate(rfecv.get_support()):
		if val:
			print "RFE - Choosing feature: " + feature_names[idx]
			result.append(feature_names[idx])
	return result


def univariate_features_with_mi(data_X, data_Y, feature_names):
	result = []

	selector = SelectPercentile(mutual_info_classif, percentile=25)
	selector.fit(data_X, data_Y)

	for idx, val in enumerate(selector.get_support()):
		if val:
			result.append(feature_names[idx])
			print "MI - Choosing feature: " + feature_names[idx]

	return result


def select_features_with_rfe_with_stratified_k_fold(data_X, data_Y, feature_names):
	result = []

	svc = SVC(kernel="linear", C=1)
	rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
	rfecv.fit(data_X, data_Y)

	print("RFE stratified_k - Optimal number of features : %d" % rfecv.n_features_)

	for idx, val in enumerate(rfecv.get_support()):
		if val:
			print "RFE stratified_k - Choosing feature: " + feature_names[idx]
			result.append(feature_names[idx])
	return result


def univariate_features_with_f_classif(data_X, data_Y, feature_names):
	result = []

	selector = SelectPercentile(f_classif, percentile=25)
	selector.fit(data_X, data_Y)

	for idx, val in enumerate(selector.get_support()):
		if val:
			result.append(feature_names[idx])
			print "f-classif - Choosing feature: " + feature_names[idx]

	return result


def univariate_features_with_f_classif(data_X, data_Y, feature_names):
	result = []

	selector = SelectPercentile(f_classif, percentile=25)
	selector.fit(data_X, data_Y)

	for idx, val in enumerate(selector.get_support()):
		if val:
			result.append(feature_names[idx])
			print "f-classif - Choosing feature: " + feature_names[idx]

	return result


def embedded_features_by_descision_tree(data_X, data_Y, feature_names):
	result = []

	clf = ExtraTreesClassifier()
	clf = clf.fit(data_X, data_Y)
	tree_weights = clf.feature_importances_
	tree_weights /= tree_weights.max()
	tree_booleans = tree_weights > np.percentile(tree_weights, 75)
	for idx, val in enumerate(tree_booleans):
		if val:
			result.append(feature_names[idx])
			print "Tree Clasifier - Choosing feature: " + feature_names[idx]

	return result


def main():
	redundant_features = []
	useful_features = []
	dic = {}

	df = import_data()

	all_features, categorical_features, numeric_features = group_features(df, dic)
	not_needed_features = fill_numeric_by_correlation(df, 0.95, numeric_features)
	fill_numeric_by_correlation(df, 0.8, numeric_features)
	print "# Added not_needed_features to redundant features"
	redundant_features.extend(not_needed_features)

	fill_numeric_features(df, numeric_features)
	fill_categorical_features(df, categorical_features)
	# transform_categorical_features(df, categorical_features)  # We Don't need that!!
	transform_label(df, "Vote")
	transform_manual(df, dic)

	scale_numeric(df, numeric_features)
	print "Before outliar detacction: " + str(df.shape[0])
	df = outliar_detection(df, numeric_features)
	print "After outliar detacction: " + str(df.shape[0])

	print "## redundant features"
	print redundant_features
	df_no_NAN = df.drop(redundant_features, axis=1).dropna()

	df_data_X, df_data_Y, features_list = to_np_array(df_no_NAN)
	df_data_X = preprocessing.scale(df_data_X)

	features_to_exclude = variance_filter(df_data_X, features_list)
	redundant_features.extend(features_to_exclude)

	good_features = select_features_with_rfe(df_data_X, df_data_Y, features_list)
	print '# ADDED Features'
	print list(set(good_features).difference(useful_features))
	useful_features.extend(good_features)

	good_features = select_features_with_rfe_with_stratified_k_fold(df_data_X, df_data_Y, features_list)
	print '# ADDED Features'
	print list(set(good_features).difference(useful_features))
	useful_features.extend(good_features)

	good_features = univariate_features_with_mi(df_data_X, df_data_Y, features_list)
	print '# ADDED Features'
	print list(set(good_features).difference(useful_features))
	useful_features.extend(good_features)

	good_features = univariate_features_with_f_classif(df_data_X, df_data_Y, features_list)
	print '# ADDED Features'
	print list(set(good_features).difference(useful_features))
	useful_features.extend(good_features)

	good_features = embedded_features_by_descision_tree(df_data_X, df_data_Y, features_list)
	print '# ADDED Features'
	print list(set(good_features).difference(useful_features))
	useful_features.extend(good_features)

	useful_features = list(set(useful_features))
	print "### Useful features ###"
	print useful_features

	print "### Redundant features ###"
	print redundant_features

	print "### Final features ###"
	print list(set(useful_features).difference(redundant_features))

	export_transformed_data(df[useful_features + ['Vote', 'split']])

	base_feature = map(lambda x: dic[x], useful_features)
	print "### Base features ###"
	print list(set(base_feature))

	embed()


if __name__ == '__main__':
	main()

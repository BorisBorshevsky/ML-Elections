import pandas as pd

import numpy as np
from IPython import embed
from sklearn import clone
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

models = {
	"SVC Linear kernel": SVC(kernel='linear'),
	"linearSVC OVR": LinearSVC(multi_class='ovr'),
	"linearSVC crammer_singer": LinearSVC(multi_class='crammer_singer'),
	"One Vs One": OneVsOneClassifier(LinearSVC()),
	"Naive Bayes": GaussianNB(),
	"Perceptron": Perceptron(max_iter=300),
	"LinearDiscriminantAnalysis": LinearDiscriminantAnalysis()
}


def add_parametrized_models():
	for splitter in range(2, 20):
		models["DecisionTreeClassifier with splitter %d" % splitter] = DecisionTreeClassifier(min_samples_split=splitter,
		                                                                                      random_state=0)
	for splitter in range(2, 20):
		models["RandomForestClassifier with splitter %d" % splitter] = RandomForestClassifier(min_samples_split=splitter,
		                                                                                      random_state=0)
	for n in range(2, 20):
		models["KNeighborsClassifier with n=%d" % n] = KNeighborsClassifier(n_neighbors=n)


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
	labels = ["Blues", "Browns", "Greens", "Greys", "Oranges", "Pinks", "Purples", "Reds", "Whites", "Yellows"]
	return df_train_X, df_train_Y, df_test_X, df_test_Y, labels


def evaluate_and_get_best(features, target):
	max_model = "linearSVC crammer_singer"
	max_score = 0
	for k, v in models.iteritems():
		scores = cross_val_score(v, features, target, cv=15)
		score = np.mean(scores)
		print "%s - Score: %f" % (k, score)
		if score > max_score:
			max_score = score
			max_model = k
	return max_model


def main():
	df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()

	train_val_data = pd.concat([df_train_X])
	features = train_val_data.values
	target = pd.concat([df_train_Y]).values

	add_parametrized_models()
	best_model_name = evaluate_and_get_best(features, target)

	clf = clone(models[best_model_name])
	clf.fit(df_test_X, df_test_Y)

	print "#######################"
	print "Prediction"
	print "#######################"

	pred = clf.predict(df_test_X)

	distribution = np.bincount(pred.astype('int64'))

	for index, party in enumerate(distribution):
		print "%s, %f, %f" % (labels[index], distribution[index], distribution[index] / float(df_test_Y.size) * 100) + '%'


if __name__ == '__main__':
	main()

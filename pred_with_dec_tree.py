import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier


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


def main():
	df_train_X, df_train_Y, df_test_X, df_test_Y, labels = load_prepared_data()

	train_val_data = pd.concat([df_train_X])
	features = train_val_data.values
	target = pd.concat([df_train_Y]).values

	clf = DecisionTreeClassifier(min_samples_split=4, random_state=0)
	clf.fit(features, target)

	features_test = df_test_X
	target_test = df_test_Y

	pred = clf.predict(features_test)

	target_test_labled = target_test.map(lambda x: labels[int(x)])
	pred_test_labled = pd.DataFrame(pred).applymap(lambda x: labels[int(x)])

	distribution = np.bincount(pred.astype('int64'))
	most_common = np.argmax(distribution)

	print "winner is party ## %s ##" % labels[most_common.astype('int')]
	print "Parties vote distribution"
	distribution = np.bincount(pred.astype('int64'))

	for index, party in enumerate(distribution):
		print "%s, %f, %f" % (
			labels[index], distribution[index], distribution[index] / float(target_test.size) * 100) + '%'

	pred_test_labled.to_csv("./data/output/test_predictions_dec_tree.csv", header=['Vote'], index=False)


print "#######################"
print "Prediction"
print "#######################"

if __name__ == '__main__':
	main()

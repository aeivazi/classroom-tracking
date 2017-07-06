import os
import argparse
import pandas as pd
from sklearn.svm import SVC
import cPickle as pickle


def read_data(model_dir):
    features_path = os.path.join(model_dir, 'features.csv')
    features_df = pd.read_csv(features_path)

    labels_path = os.path.join(model_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_path)

    return features_df, labels_df


def create_classifier(features, labels):

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(features, labels)
    return clf


def main(args):
    features_df, labels_df = read_data(args.model_dir)
    features = features_df.as_matrix()

    labels_df1 = labels_df.ix[:,0]
    labels = labels_df1.as_matrix()

    clf = create_classifier(features, labels)

    model_path = os.path.join(args.model_dir, 'model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(clf, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='Path to features and labels csv')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
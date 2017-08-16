import os
import argparse
import pandas as pd
from matplotlib import pyplot as plt
from tsne import bh_sne

def read_data(model_dir):
    features_path = os.path.join(model_dir, 'features.csv')
    features_df = pd.read_csv(features_path, header=None)

    labels_path = os.path.join(model_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_path)

    return features_df, labels_df


def main(args):
    features_df, labels_df = read_data(args.model_dir)
    X = features_df.as_matrix()

    labels_df = labels_df.ix[:, 1]
    y = labels_df.as_matrix()

    X_2d = bh_sne(X, perplexity=5)

    # plot the result
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, marker='s')
    plt.colorbar(ticks=range(10))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='Path to features and labels csv')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
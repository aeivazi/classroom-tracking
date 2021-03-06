#!/usr/bin/env python2
#
# Tool to calculate representation value.
# Heavily based on openface.
#
# Anna Eivazi
# 2017/06/21
#

import argparse
import os
import pandas as pd
from sklearn.svm import SVC
import cPickle as pickle

import openface

from src.image_reader import get_image_list
from src.features_calculator import image_to_features


def do_talking(message):
    # just a helper function to make code more readable
    if args.verbose:
        print(message)


def parent_folder(path):
    return os.path.basename(os.path.dirname(path))


def create_classifier(features, labels):

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(features, labels)
    return clf


def calculate_features(images, model, output_path, verbose=False):

    feature_values = [image_to_features(img, model, verbose) for img in images]

    features_df = pd.DataFrame(feature_values)
    features_df.to_csv(output_path, index=False, header=False)

    return features_df


def calculate_labels(images, output_path):

    participants_set = {parent_folder(image) for image in images}
    participant_as_index = {p: ind for ind, p in enumerate(participants_set)}

    images_with_ind = [[participant_as_index[parent_folder(image)], parent_folder(image)] for image in images]

    labels_df = pd.DataFrame(images_with_ind, columns=['numeric', 'text'])
    labels_df.to_csv(output_path)

    return labels_df


def calculate_svm_model(features_df, labels_df, output_dir):
    features = features_df.as_matrix()

    labels_only_df = labels_df.ix[:, 'numeric']
    labels = labels_only_df.as_matrix()

    clf = create_classifier(features, labels)

    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(clf, model_file)


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = openface.TorchNeuralNet(args.networkModel, args.img_dim)

    print('Start processing...')

    if not (os.path.exists(args.img_dir)):
        raise ValueError('Input directory does not exist: {}'.format(args.img_dir))

    images = get_image_list(args.img_dir)
    do_talking('Found {} images to process'.format(len(images)))

    features_path = os.path.join(args.out_dir, 'features.csv')
    features_df = calculate_features(images, model, features_path, args.verbose)

    labels_path = os.path.join(args.out_dir, 'labels.csv')
    labels_df = calculate_labels(images, labels_path)

    calculate_svm_model(features_df, labels_df, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str, help='Directory with aligned images')
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('--networkModel', type=str, help='Path to Torch network model.',
                        default=os.path.join('/home/anna/src/openface/models/openface', 'nn4.small2.v1.t7'))
    parser.add_argument('--img-dim', type=int,help='Default image dimension.', default=96)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
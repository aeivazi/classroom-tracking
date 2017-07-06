#!/usr/bin/env python2
#
# Tool to calculate representation value.
# Heavily based on openface.
#
# Anna Eivazi
# 2017/06/21
#

import time
import argparse
import os
import pandas as pd

import openface

from src.image_reader import get_image_list
from src.features_calculator import image_to_features

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('img_dir', type=str, help='Directory with aligned images')
parser.add_argument('out_dir', type=str, help='Output directory')
parser.add_argument('--networkModel', type=str, help='Path to Torch network model.',
                    default=os.path.join('/home/anna/src/openface/models/openface', 'nn4.small2.v1.t7'))
parser.add_argument('--img-dim', type=int,help='Default image dimension.', default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()


def do_talking(message):
    # just a helper function to make code more readable
    if args.verbose:
        print(message)


def parent_folder(path):
    return os.path.basename(os.path.dirname(path))

do_talking('Argument parsing and loading libraries took {} seconds.'.format(time.time() - start))

start = time.time()
model = openface.TorchNeuralNet(args.networkModel, args.img_dim)
do_talking('Loading the OpenFace model and align library took {} seconds.'.format(time.time() - start))

print('Start processing...')

if not (os.path.exists(args.img_dir)):
    raise ValueError('Input directory does not exist: {}'.format(args.img_dir))

images = get_image_list(args.img_dir)
do_talking('Found {} images to process'.format(len(images)))

feature_values = [image_to_features(img, model, args.verbose) for img in images]
features_df = pd.DataFrame(feature_values)
features_df.to_csv(os.path.join(args.out_dir, 'features.csv'), index=False, header=False)

participants_set ={parent_folder(image) for image in images}
participant_as_index = {p : ind for ind, p in enumerate(participants_set)}

images_with_ind = [[participant_as_index[parent_folder(image)],image] for image in images]

labels_df = pd.DataFrame(images_with_ind)
labels_df.to_csv(os.path.join(args.out_dir, 'labels.csv'), index=False, header=False)




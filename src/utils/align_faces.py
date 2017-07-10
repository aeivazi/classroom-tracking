#!/usr/bin/env python2
#
# Aligns the biggest face on the image by using openface utilities.
#
# Anna Eivazi
# 2017/06/28
#

import argparse
import cv2
import os
import openface

from src.image_reader import get_image_list
from src.face_aligner import calculate_aligned_face


def main(args):

    aligner = openface.AlignDlib(args.dlibFacePredictor)

    if not (os.path.exists(args.img_dir)):
        raise ValueError('Input directory does not exist: {}'.format(args.img_dir))

    images = get_image_list(args.img_dir)
    if args.verbose:
        print('Found {} images to process'.format(len(images)))

    for img_path in images:

        try:
            aligned_face = calculate_aligned_face(img_path, aligner, args.img_dim, args.verbose)
        except ValueError as err:
            print(err.message)
            continue

        if aligned_face is not None:

            aligned_path = os.path.join(args.out_dir, os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))
            if args.verbose:
                print('Writing aligned image to {}'.format(aligned_path))

            this_dir = os.path.dirname(aligned_path)
            if not os.path.exists(this_dir):
                os.makedirs(this_dir)

            cv2.imwrite(aligned_path, aligned_face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('img_dir', type=str, help='Input images directory.')
    parser.add_argument('out_dir', type=str, help='Output directory for aligned faces.')
    parser.add_argument('--dlibFacePredictor', type=str, help='Path to dlibs face predictor.',
                        default=os.path.join(
                            '/home/anna/src/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--img-dim', type=int, help='Default image dimension.', default=96)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
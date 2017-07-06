import os
import argparse

import openface

import time

start = time.time()

from src.xml_parser import read_crowd_gaze_xml
from src.clip_face import clip_face, save_face

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-xml', help='path to main xml file with faces detected', type=str, required=True)
    parser.add_argument('--dlib-face-predictor', type=str, help="Path to dlib's face predictor.",  default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.", default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--img-dim', type=int, help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-o', '--output-faces-dir', help='path to directory where faces images will be stored', type=str, required=True)
    args = parser.parse_args()
    return args


def align(img_path):
    if args.verbose:
        print("Processing {}.".format(img_path))
    bgrImg = cv2.imread(img_path)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(img_path))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

if __name__ == "__main__":
    args = parse_arguments()

    if args.verbose:
        print("Argument parsing and loading libraries took {} seconds.".format(time.time() - start))

    start = time.time()
    align = openface.AlignDlib(args.dlib_face_predictor)
    net = openface.TorchNeuralNet(args.networkModel, args.img_dim)
    if args.verbose:
        print("Loading the OpenFace model and align library took {} seconds.".format(time.time() - start))

    images = read_crowd_gaze_xml(args.input_xml)

    faces_dir = os.path.dirname(args.output_faces_dir)
    if not os.path.exists(faces_dir):
        os.mkdir(faces_dir)

    image_dir = os.path.dirname(args.input_xml)
    for image in images:
        for participant in image['participants']:
            face_image = clip_face(image_path_absolute = os.path.join(image_dir, image['path']),
                                   width=participant['face_box']['width'],
                                   height=participant['face_box']['height'],
                                   top=participant['face_box']['top'],
                                   left=participant['face_box']['left'],
                                   expand_by=0)
            save_face(face_image, faces_dir, image['path'], participant['label'])






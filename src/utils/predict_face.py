import os
import argparse
import cPickle as pickle
import openface

from src.features_calculator import calculate_features
from src.face_aligner import calculate_aligned_face
from src.face_classification_predictor import get_labels_dict, predict_face


def main(args):
    face_classification_model = pickle.load(open(os.path.join(args.model_dir, 'model.pkl'), 'rb'))

    aligner = openface.AlignDlib(args.dlibFacePredictor)
    aligned_face = \
        calculate_aligned_face(img_path=args.image_to_predict, aligner=aligner, img_dim=args.img_dim, verbose=args.verbose)

    features_model = openface.TorchNeuralNet(args.networkModel, args.img_dim)

    labels_dict = get_labels_dict(os.path.join(args.model_dir, 'labels.csv'))

    face_features = calculate_features(aligned_face, features_model)

    prediction = predict_face(face_features, face_classification_model, labels_dict)

    print(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_to_predict', type=str, help='Path to image to predict')
    parser.add_argument('model_dir', type=str, help='Path to model directory')
    parser.add_argument('--networkModel', type=str, help='Path to Torch network model.',
                        default=os.path.join('/home/anna/src/openface/models/openface', 'nn4.small2.v1.t7'))
    parser.add_argument('--dlibFacePredictor', type=str, help='Path to dlibs face predictor.',
                        default=os.path.join(
                            '/home/anna/src/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--img-dim', type=int, help='Default image dimension.', default=96)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
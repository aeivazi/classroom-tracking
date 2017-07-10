import os
import argparse
import cPickle as pickle
import openface
import cv2


from src.xml_parser import read_crowd_gaze_xml, write_crowd_gaze_xml
from src.features_calculator import calculate_features
from src.face_aligner import calculate_aligned_face_from_mat
from src.face_clipper import clip_matrix


def main(args):
    xml_tree = read_crowd_gaze_xml(args.input_xml)

    face_classification_model = pickle.load(open(os.path.join(args.model_dir, 'model.pkl'), 'rb'))
    face_aligner = openface.AlignDlib(args.dlibFacePredictor)
    features_model = openface.TorchNeuralNet(args.networkModel, args.img_dim)

    image_dir = os.path.dirname(args.input_xml)

    for ind, image in enumerate(xml_tree.getroot().iter('image')):

        # stop processing if user defined it so
        if args.frames_end and ind > args.frames_end:
            break

        if args.verbose:
            print('Processing image {}'.format(image['path']))

        image_path = os.path.join(image_dir, image.attrib['file'])
        print(image_path)
        img = cv2.imread(image_path)

        for box in image.iter('box'):

            face = clip_matrix(img,
                               width=int(box.attrib['width']),
                               height=int(box.attrib['height']),
                               top=int(box.attrib['top']),
                               left=int(box.attrib['left']),
                               expand_by=10)

            aligned_face = calculate_aligned_face_from_mat(face, face_aligner)

            image_as_features = calculate_features(aligned_face, features_model)

            prediction = face_classification_model.predict(image_as_features)

            box.find('label').text = prediction

    write_crowd_gaze_xml(xml_tree, args.output_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml', help='path to an xml file with faces detected', type=str)
    parser.add_argument('output_xml', help='path to output xml', type=str)
    parser.add_argument('model_dir', help='path to model directory with pickle and labels cvs', type=str)
    parser.add_argument('-fe', '--frames-end', help='process till this frame', type=int, required=False)
    parser.add_argument('--networkModel', type=str, help='Path to Torch network model.',
                        default=os.path.join('/home/anna/src/openface/models/openface', 'nn4.small2.v1.t7'))
    parser.add_argument('--dlibFacePredictor', type=str, help='Path to dlibs face predictor.',
                        default=os.path.join(
                            '/home/anna/src/openface/models/dlib/shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--img-dim', type=int, help='Default image dimension.', default=96)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)






import os
import argparse
import cPickle as pickle
import openface
import cv2


from src.xml_parser import read_crowd_gaze_xml, write_crowd_gaze_xml, read_eyes_coord
from src.features_calculator import calculate_features
from src.face_aligner import align
from src.face_clipper import clip_matrix
from src.face_classification_predictor import predict_face, get_labels_dict


def main(args):

    # read xml as a tree
    xml_tree = read_crowd_gaze_xml(args.input_xml)

    # read the openface model for converting image to 128 features
    features_model = openface.TorchNeuralNet(args.networkModel, args.img_dim)

    # read a model for face classification
    face_classification_model = pickle.load(open(os.path.join(args.model_dir, 'model.pkl'), 'rb'))

    # read labels
    labels_dict = get_labels_dict(os.path.join(args.model_dir, 'labels.csv'))

    # image directory is the same where input xml is
    image_dir = os.path.dirname(args.input_xml)

    # process for each frame
    for ind, image in enumerate(xml_tree.getroot().iter('image')):

        # stop processing if user defined it so
        if args.frames_end and ind >= args.frames_end:
            break

        if args.verbose:
            print('Processing image {}'.format(image.attrib['file']))

        image_path = os.path.join(image_dir, image.attrib['file'])

        # read image as RGB
        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # process every face
        for box in image.iter('box'):

            face = clip_matrix(rgb_img,
                               width=int(box.attrib['width']),
                               height=int(box.attrib['height']),
                               top=int(box.attrib['top']),
                               left=int(box.attrib['left']),
                               expand_by=0)
            try:
                left_eye, right_eye = read_eyes_coord(box)
                aligned_face = align(face, left_eye, right_eye, args.img_dim)
            except ValueError:
                print('Could not align face, cannot continue with prediction. Label with be set to Unknown')
                box.find('label').text='Unknown'
                continue

            # calculate features for aligned_face
            face_features = calculate_features(aligned_face, features_model)

            # predict label
            prediction = predict_face(face_features, face_classification_model, labels_dict)
            box.find('label').text = str(prediction)

    # write new xml tree to output file
    write_crowd_gaze_xml(xml_tree, args.output_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml', help='path to an xml file with faces detected', type=str)
    parser.add_argument('output_xml', help='path to output xml', type=str)
    parser.add_argument('model_dir', help='path to model directory with pickle and labels cvs', type=str)
    parser.add_argument('-fe', '--frames-end', help='process till this frame', type=int, required=False)
    parser.add_argument('--networkModel', type=str, help='Path to Torch network model.',
                        default=os.path.join('/home/anna/src/openface/models/openface', 'nn4.small2.v1.t7'))
    parser.add_argument('--img-dim', type=int, help='Default image dimension.', default=96)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)







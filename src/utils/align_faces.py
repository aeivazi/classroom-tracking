import os
import argparse
import cv2
import copy

from src.xml_parser import read_crowd_gaze_xml, read_eyes_coord
from src.face_clipper import clip_matrix
from src.face_aligner import align


def save_image(im, dir_path, image_name):
    image_path = os.path.join(dir_path, image_name)
    image_matrix_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_matrix_bgr)


def main(args):
    xml_tree = read_crowd_gaze_xml(args.input_xml)

    output_dir = os.path.dirname(args.output_faces_dir)

    image_dir = os.path.dirname(args.input_xml)
    for ind, image in enumerate(xml_tree.getroot().iter('image')):

        # skip frames if user defined it so
        if args.frames_start and ind <= args.frames_start:
            continue

        # stop processing if user defined it so
        if args.frames_end and ind > args.frames_end:
            break

        if args.verbose:
            print('Processing image {}'.format(image.attrib['file']))

        # read image as rgb
        image_path = os.path.join(image_dir, image.attrib['file'])
        image_matrix_bgr = cv2.imread(image_path)
        image_matrix_rgb = cv2.cvtColor(image_matrix_bgr, cv2.COLOR_BGR2RGB)

        # make frame directory
        frame_name = os.path.splitext(os.path.basename(image.attrib['file']))[0]
        frame_dir = os.path.join(output_dir, frame_name)
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)

        for box in image.iter('box'):

            # clip a face from the frame image
            face = clip_matrix(image_matrix_rgb,
                               width=int(box.attrib['width']),
                               height=int(box.attrib['height']),
                               top=int(box.attrib['top']),
                               left=int(box.attrib['left']),
                               expand_by=0)

            # run face alignment algorithm:
            #  1. make eyes line horizontal
            #  2. scale the face to 96x96

            # read eye landmarks from xml
            left_eye, right_eye = read_eyes_coord(box)

            # run alignment and save the aligned face
            aligned_face = align(face, left_eye, right_eye, args.desired_scaled_dim)
            aligned_image_name = '{}.jpg'.format(box.find('label').text)
            save_image(aligned_face, frame_dir, aligned_image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml', help='path to an xml file with faces detected', type=str)
    parser.add_argument('output_faces_dir', help='path to directory where faces images will be stored', type=str)
    parser.add_argument('--desired-scaled-dim', help='desired scaled dimention of the output image', type=int, default=96)
    parser.add_argument('-fs', '--frames-start', help='process starting from this frame', type=int, required=False)
    parser.add_argument('-fe', '--frames-end', help='process till this frame', type=int, required=False)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)







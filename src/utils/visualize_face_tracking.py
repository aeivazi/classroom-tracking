import os
import argparse
import numpy as np
import cv2


from src.xml_parser import read_crowd_gaze_xml, read_eyes_coord
from src.face_aligner import align
from src.face_clipper import clip_matrix


def main(args):

    # read xml as a tree
    xml_tree = read_crowd_gaze_xml(args.input_xml)

    # image directory is the same where input xml is
    image_input_dir = os.path.dirname(args.input_xml)

    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # process for each frame
    for ind, image in enumerate(xml_tree.getroot().iter('image')):

        # stop processing if user defined it so
        if args.frames_end and ind >= args.frames_end:
            break

        if args.verbose:
            print('Processing image {}'.format(image.attrib['file']))

        image_path = os.path.join(image_input_dir, image.attrib['file'])

        # read image as RGB
        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # process every face
        for box in image.iter('box'):

            width = int(box.attrib['width'])
            height = int(box.attrib['height'])
            top = int(box.attrib['top'])
            left = int(box.attrib['left'])

            cv2.rectangle(rgb_img, (left, top), (left + width, top + height), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_img, box.find('label').text, (left, top), font, 1, (255, 0, 0), 2)

        image_output_path = os.path.join(args.output_dir, image.attrib['file'])
        image_as_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_output_path, image_as_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml', help='path to an xml file with face that were tracked already', type=str)
    parser.add_argument('output_dir', help='path to directory where new images will be saved', type=str)
    parser.add_argument('-fe', '--frames-end', help='process till this frame', type=int, required=False)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)







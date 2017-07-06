import os
import argparse


from src.xml_parser import read_crowd_gaze_xml
from src.clip_face import clip_face


def save_face(im, dir_path, original_image_path, participant_label):

    image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    face_dir_path = os.path.join(dir_path, image_name)

    if not os.path.exists(face_dir_path):
        os.makedirs(face_dir_path)

    face_path = os.path.join(face_dir_path, '{}.jpg'.format(participant_label))
    im.save(face_path)


def main(args):
    images = read_crowd_gaze_xml(args.input_xml)

    faces_dir = os.path.dirname(args.output_faces_dir)
    # if not os.path.exists(faces_dir):
    #     os.mkdir(faces_dir)

    image_dir = os.path.dirname(args.input_xml)
    for ind, image in enumerate(images):

        #stop processing if limit is set
        if args.frames_limit and ind > args.frames_limit:
            break

        if args.verbose:
            print('Processing image {}'.format(image['path']))

        for participant in image['participants']:
            face_image = clip_face(image_path_absolute=os.path.join(image_dir, image['path']),
                                   width=participant['face_box']['width'],
                                   height=participant['face_box']['height'],
                                   top=participant['face_box']['top'],
                                   left=participant['face_box']['left'],
                                   expand_by=5)
            save_face(face_image, faces_dir, image['path'], participant['label'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml', help='path to an xml file with faces detected', type=str)
    parser.add_argument('-o', '--output-faces-dir', help='path to directory where faces images will be stored',
                        type=str, required=True)
    parser.add_argument('-fl', '--frames-limit', help='process just first n frames', type=int, required=False)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)







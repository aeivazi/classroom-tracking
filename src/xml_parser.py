import xml.etree.ElementTree as ET


def read_crowd_gaze_xml(xml_file_path):
    """
    Reads xml and return the whole tree
    """
    return ET.parse(xml_file_path)


def write_crowd_gaze_xml(tree, xml_file_path):
    """
    Writes tree to xml file
    """
    tree.write(xml_file_path)


def read_eyes_coord(box):
    """
    Reads eyes landmarks and recalculated eye coordinates to originate from the box upper left corner.

    :param box: xml box item, that should have a structure as crowd gaze xml dictates
    :return: left and right eyes coordinates
    """
    left_eye = None
    right_eye = None
    for landmark in box.iter('point'):

        # we are only interested in eye landmarks
        if landmark.attrib['idx'] != '1' and landmark.attrib['idx'] != '2':
            continue

        # integer precision is just good enough -> pixel precision
        landmark_x = int(float(landmark.attrib['x']))
        landmark_y = int(float(landmark.attrib['y']))

        # recalculate landmarks coordinates relatively to box upper left corner
        landmark_x -= int(box.attrib['left'])
        landmark_y -= int(box.attrib['top'])

        if landmark.attrib['idx'] == '1':
            right_eye = (landmark_x, landmark_y)

        if landmark.attrib['idx'] == '2':
            left_eye = (landmark_x, landmark_y)

    return left_eye, right_eye

import xml.etree.ElementTree as ET


"""Reads xml and return list of images.

   Structure of the image is:
   image: {'path': str, 'participants': list}
   participant: {'label': str, 'face_box': dict}
   face_box: {'height': int, 'width': int, 'left': int, 'top': int}
"""
def read_crowd_gaze_xml(xml_file_path):

    dataset = ET.parse(xml_file_path).getroot()

    images = []

    for image in dataset.iter('image'):

        participants = []
        for box in image.iter('box'):
            face_box = {'height': int(box.attrib['height']),
                        'width': int(box.attrib['width']),
                        'left': int(box.attrib['left']),
                        'top': int(box.attrib['top'])}

            participants.append({'label': box.find('label').text, 'face_box': face_box})

        images.append({'path': image.attrib['file'], 'participants': participants})

    return images

import os
from PIL import Image


def clip_face(image_path_absolute, width, height, top, left, expand_by=0):

    im = Image.open(image_path_absolute)

    #PIL expect box in form (left, upper, right, lower)
    pil_box = (left, top, left + width, top + height)

    #expand bounding box to every direction
    pil_box = (pil_box[0]-expand_by, pil_box[1]-expand_by, pil_box[2]+expand_by, pil_box[3]+expand_by)

    im = im.crop(pil_box)

    return im





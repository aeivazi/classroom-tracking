import os


def get_image_list(pathDir):
    """
    Reads all jpg in root folder and subfolder.
    Returns list of images as a full path.
    """
    images = []
    for (dirpath, dirnames, filenames) in os.walk(pathDir):
        for f in filenames:
            extension = os.path.basename(f).split('.')[1]
            if extension.lower() == 'jpg':
                imagePath = os.path.join(dirpath, f)
                images.append(imagePath)
    return images
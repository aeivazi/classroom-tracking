import cv2


def image_to_features(img_path, model, verbose=False):
    """
    Calculates 128 features for an image using given model.
    """
    if verbose:
        print('Processing {}.'.format(img_path))

    bgrImg = cv2.imread(img_path)
    if bgrImg is None:
        raise Exception('Unable to load image: {}'.format(img_path))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    return calculate_features(rgbImg, model)


def calculate_features(cv2_image, model):
    """
    Calculates 128 features for an given image as a cv2 matrix.
    """
    rep = model.forward(cv2_image)
    return rep

import cv2


def calculate_aligned_face(img_path, aligner, img_dim=96, verbose=False):
    if verbose:
        print('Processing {}.'.format(img_path))

    bgrImg = cv2.imread(img_path)
    if bgrImg is None:
        raise ValueError('Unable to load image: {}'.format(img_path))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if verbose:
        print('  + Original size: {}'.format(rgbImg.shape))

    bb = aligner.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise ValueError('Unable to find a face: {}'.format(img_path))

    aligned_face = aligner.align(img_dim, rgbImg, bb, landmarkIndices=aligner.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise ValueError('Unable to align image: {}'.format(img_path))

    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

    return aligned_face
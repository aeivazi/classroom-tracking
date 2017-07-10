import cv2


def calculate_aligned_face(img_path, aligner, img_dim=96, verbose=False):
    if verbose:
        print('Processing {}.'.format(img_path))

    bgrImg = cv2.imread(img_path)
    if bgrImg is None:
        raise ValueError('Unable to load image: {}'.format(img_path))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    return calculate_aligned_face_from_mat(rgbImg, aligner, img_dim, verbose)


def calculate_aligned_face_from_mat(img_as_matrix, aligner, img_dim=96, verbose=False):

    if verbose:
        print('Original size: {}'.format(img_as_matrix.shape))

    bb = aligner.getLargestFaceBoundingBox(img_as_matrix)
    if bb is None:
        raise ValueError('Unable to find a face')

    aligned_face = aligner.align(img_dim, img_as_matrix, bb, landmarkIndices=aligner.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise ValueError('Unable to align image')

    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

    return aligned_face
import cv2
import numpy as np


def calculate_aligned_face(img_path, aligner, img_dim=96, verbose=False):
    if verbose:
        print('Processing {}.'.format(img_path))

    bgrImg = cv2.imread(img_path)
    if bgrImg is None:
        raise ValueError('Unable to load image: {}'.format(img_path))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    return calculate_aligned_face_from_mat(rgbImg, aligner, img_dim, verbose)


def align(image_as_mat, left_eye_center, right_eye_center, desired_dim, desired_left_eye=(0.35, 0.35)):
    """
    Rotates and scales input image.
    The rotation angle is calculated so that the line between left_eye_center and right_eye_center will become horizontal.
    The scaling is calculated so that left eye would located on the place as desired_left_eye dictates and
    the output size would be 96x96 pixels.

    :param image_as_mat: opencv like matrix of the image
    :param left_eye_center: coordinates of the left eye on the image
    :param right_eye_center: coordinates of the right eye on the image
    :param desired_left_eye: the relative desired position of the left eye
    :param desired_dim: the desired dimension of the output matrix
    :return: aligned matrix
    """

    # compute the angle between the eye centroids
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desired_left_eye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desired_left_eye[0])
    desiredDist *= desired_dim
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desired_dim * 0.5
    tY = desired_dim * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desired_dim, desired_dim)
    output = cv2.warpAffine(image_as_mat, M, (w, h), flags=cv2.INTER_CUBIC)

    return output
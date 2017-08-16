import cv2
import numpy as np

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

LEFT_EYE_SRC = 0
RIGHT_EYE_SRC = 1
LEFT_MOUTH_CORNER_SRC = 3
RIGHT_MOUTH_CORNER_SRC = 4

LEFT_EYE_CORNERS_DST = [36, 39]
RIGHT_EYE_CORNERS_DST = [42, 45]
LEFT_MOUTH_CORNER_DST = 48
RIGHT_MOUTH_CORNER_DST = 54


def align(image_as_mat, landmarks, desired_dim):
    """
    Aligns face by three points: right eye, left eye and center of mouth.
    Runs affine transformation to align three points according predefined places as openface uses it (see MINMAX_TEMPLATE)

    :param image_as_mat: opencv like matrix of the image
    :param landmarks: coordinates of five landmarks
    :param desired_dim: the desired dimension of the output matrix
    :return: aligned matrix
    """

    left_eye_src = landmarks[LEFT_EYE_SRC]
    right_eye_src = landmarks[RIGHT_EYE_SRC]
    mounth_center_src = np.mean([landmarks[LEFT_MOUTH_CORNER_SRC], landmarks[RIGHT_MOUTH_CORNER_SRC]], axis=0)
    landmarks_src = np.float32([left_eye_src, right_eye_src, mounth_center_src])

    left_eye_dst = np.mean(TEMPLATE[LEFT_EYE_CORNERS_DST], axis=0)
    right_eye_dst = np.mean(TEMPLATE[RIGHT_EYE_CORNERS_DST], axis=0)
    mouth_center_dst = np.mean([TEMPLATE[LEFT_MOUTH_CORNER_DST], TEMPLATE[RIGHT_MOUTH_CORNER_DST]], axis=0)
    landmarks_dst = desired_dim*np.float32([left_eye_dst, right_eye_dst, mouth_center_dst])

    M = cv2.getAffineTransform(landmarks_src, landmarks_dst)

    output = cv2.warpAffine(image_as_mat, M, (desired_dim, desired_dim), flags=cv2.INTER_CUBIC)

    return output


def align_eyes_only(image_as_mat, landmarks, desired_dim):
    """
    Rotates and scales input image.
    The rotation angle is calculated so that the line between left_eye_center and right_eye_center will become horizontal.
    The scaling is calculated so that left eye would located on the place as desired_left_eye dictates and
    the output size would be 96x96 pixels.

    :param image_as_mat: opencv like matrix of the image
    :param landmarks: coordinates of five landmarks
    :param desired_dim: the desired dimension of the output matrix
    :param desired_left_eye: the relative desired position of the left eye
    :return: aligned matrix
    """

    desired_left_eye = np.mean(TEMPLATE[LEFT_EYE_CORNERS_DST], axis=0)
    desired_right_eye = np.mean(TEMPLATE[RIGHT_EYE_CORNERS_DST], axis=0)

    left_eye_center = landmarks[LEFT_EYE_SRC]
    right_eye_center = landmarks[RIGHT_EYE_SRC]

    # compute the angle between the eye centroids
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desired_right_eye[0] - desired_left_eye[0])
    desiredDist *= desired_dim
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((left_eye_center[0] + right_eye_center[0]) / 2,
                  (left_eye_center[1] + right_eye_center[1]) / 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desired_dim * 0.5
    tY = desired_dim * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    print(M)

    # apply the affine transformation
    output = cv2.warpAffine(image_as_mat, M, (desired_dim, desired_dim), flags=cv2.INTER_CUBIC)

    return output

def align_perspective(image_as_mat, landmarks, desired_dim):
    """
    Aligns face by four points: right eye, left eye and corners of mouth.
    Runs perspective transformation to align landmarks according predefined places as openface uses it (see MINMAX_TEMPLATE)

    :param image_as_mat: opencv like matrix of the image
    :param landmarks: coordinates of five landmarks
    :param desired_dim: the desired dimension of the output matrix
    :return: aligned matrix
    """

    landmarks_src = np.float32([landmarks[LEFT_EYE_SRC],
                                landmarks[RIGHT_EYE_SRC],
                                landmarks[LEFT_MOUTH_CORNER_SRC],
                                landmarks[RIGHT_MOUTH_CORNER_SRC]])

    left_eye_dst = np.mean(TEMPLATE[LEFT_EYE_CORNERS_DST], axis=0)
    right_eye_dst = np.mean(TEMPLATE[RIGHT_EYE_CORNERS_DST], axis=0)
    left_mouth_dist_pixel = TEMPLATE[LEFT_MOUTH_CORNER_DST]
    right_mouth_dist_pixel = TEMPLATE[RIGHT_MOUTH_CORNER_DST]

    landmarks_dst = desired_dim * np.float32([left_eye_dst,
                                              right_eye_dst,
                                              left_mouth_dist_pixel,
                                              right_mouth_dist_pixel])

    M = cv2.getPerspectiveTransform(landmarks_src, landmarks_dst)
    output = cv2.warpPerspective(image_as_mat, M, (desired_dim, desired_dim))

    return output
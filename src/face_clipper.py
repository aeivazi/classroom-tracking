
def clip_matrix(image_as_matrix, width, height, top, left, expand_by=0):

    x1 = left
    x2 = left + width
    y1 = top
    y2 = top + height

    crop_img = image_as_matrix[y1-expand_by:y2+expand_by, x1-expand_by:x2+expand_by]

    return crop_img





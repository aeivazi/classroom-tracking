

def predict_face(face_as_feature, face_classifier, labels_dict):
    """Predicts the label and returns string representation of the label."""

    raw_label = face_classifier.predict(face_as_feature.reshape(1, -1))

    string_label = labels_dict[raw_label]

    return string_label
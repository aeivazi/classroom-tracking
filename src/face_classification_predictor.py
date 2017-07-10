import pandas as pd


def predict_face(face_as_feature, face_classifier, labels_dict):
    """Predicts the label and returns string representation of the label."""

    raw_label = face_classifier.predict(face_as_feature.reshape(1, -1))

    string_label = labels_dict[raw_label[0]]

    return string_label


def get_labels_dict(labels_path):
    """Reads labels cvs and returns data as a dictionary where numeric column is key and text column is value."""
    labels_df = pd.read_csv(labels_path)
    labels_dict = labels_df.set_index('numeric')['text'].to_dict()
    return labels_dict

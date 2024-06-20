import math
from typing import List


def adjust_face_length(xs: List[float], ys: List[float], face_length: float):
    scale = math.sqrt(face_length / 0.5)
    output_ys = []
    for i in range(len(ys)):
        # index 0-16 is the face outline keypoints
        if i < 17:
            output_ys.append(0.5 + scale * (ys[i] - 0.5))
        else:
            output_ys.append(ys[i])
    return xs, output_ys


def adjust_face_width(xs: List[float], ys: List[float], face_width: float):
    scale = math.sqrt(face_width / 0.5)
    output_xs = []
    for i in range(len(xs)):
        # index 0-16 is the face outline keypoints
        if i < 17:
            output_xs.append(0.5 + scale * (xs[i] - 0.5))
        else:
            output_xs.append(xs[i])
    return output_xs, ys


def adjust_eyebrow_height(xs: List[float], ys: List[float], eyebrow_height: float):
    displacement = (eyebrow_height - 0.5) * 0.04
    output_ys = []
    for i in range(len(xs)):
        # index 17-26 is the eyebrow keypoints
        if i in range(17, 27):
            output_ys.append(ys[i] + displacement)
        else:
            output_ys.append(ys[i])
    return xs, output_ys


def adjust_eye_height(xs: List[float], ys: List[float], eye_height: float):
    displacement = (eye_height - 0.5) * 0.04
    output_ys = []
    for i in range(len(xs)):
        # index 36-47 and 68-69 is the eye keypoints
        if i in range(36, 48) or i in range(68, 70):
            output_ys.append(ys[i] + displacement)
        else:
            output_ys.append(ys[i])
    return xs, output_ys


def adjust_nose_size(xs: List[float], ys: List[float], nose_size: float):
    scale = math.sqrt(nose_size / 0.5)
    output_xs = []
    for i in range(len(xs)):
        # index 31-35 is the nose keypoints
        if i in range(31, 36):
            output_xs.append(0.5 + scale * (xs[i] - 0.5))
        else:
            output_xs.append(xs[i])
    return output_xs, ys


def adjust_mouth_size(xs: List[float], ys: List[float], mouth_size: float):
    scale = math.sqrt(mouth_size / 0.5)
    output_xs = []
    for i in range(len(xs)):
        # index 48-67 is the mouth keypoints
        if i in range(48, 68):
            output_xs.append(0.5 + scale * (xs[i] - 0.5))
        else:
            output_xs.append(xs[i])
    return output_xs, ys

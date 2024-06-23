import math
from typing import List
from PIL import Image
import csv
import numpy as np
from controlnet_aux.open_pose import draw_poses, PoseResult
from scipy.spatial.transform import Rotation as R
import cv2
from controlnet_aux.open_pose.body import Keypoint, BodyResult
from FaceOperation import *


def load_face_coordinates(filename):
    x = []
    y = []
    z = []
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if any(field.strip() for field in row):
                x.append(float(row[0]))
                y.append(float(row[1]))
                z.append(float(row[2]))
    min_z = min(z)
    max_z = max(z)
    # normalize to 0-1
    z = [(val - min_z) / (max_z - min_z) for val in z]
    print(z)
    return x, y, z


def rotate_points(points, center, angles):
    # degree to radian
    angles_rad = np.radians(angles)

    # create rotation
    rotation = R.from_euler('xyz', angles_rad)

    points_shifted = points - center

    points_rotated = rotation.apply(points_shifted)

    points_rotated += center

    return points_rotated


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def draw_facepose(canvas: np.ndarray, keypoints, eps=0.01) -> np.ndarray:
    H, W, C = canvas.shape
    for keypoint in keypoints:
        x, y = keypoint[0], keypoint[1]
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def openpose_paint(result: List[PoseResult], save_path: str):
    H = 512
    W = 512
    canvas = draw_poses(result, H, W, draw_body=True, draw_hand=True, draw_face=True)

    detected_map = canvas
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)
    detected_map.save(save_path)


def scale_up_openpose(detection_results: List[PoseResult], scale: float, save_path: str):
    face = detection_results[0].face
    body = detection_results[0].body.keypoints
    hand_l = detection_results[0].left_hand
    hand_r = detection_results[0].right_hand

    min_x, max_x, min_y, max_y = (1, 0, 1, 0)
    for keypoint in face + body + hand_r + hand_l:
        if keypoint is None or keypoint.x < 0 or keypoint.y < 0:
            continue

        if keypoint.x < min_x:
            min_x = keypoint.x
        elif keypoint.x > max_x:
            max_x = keypoint.x

        if keypoint.y < min_y:
            min_y = keypoint.y
        elif keypoint.y > max_y:
            max_y = keypoint.y

    print(min_x, max_x, min_y, max_y)
    if min_x < 0.1 or min_y < 0.1:
        return -1, -1, -1

    body_length = max_y - min_y
    body_width = max_x - min_x

    if body_width * scale > 0.85 or body_length * scale > 0.85:
        scale = min(0.85 / body_width, 0.85 / body_length)

    if scale < 1.1:
        return -1, -1, -1

    # we are moving the scaled_up min_x and min_y points to (0.1, 0.1)
    move = (0.1 - min_x * scale, 0.1 - min_y * scale)

    def mul(p: Keypoint, s: float, m: (float, float)):
        if p is None:
            return None
        else:
            return Keypoint(p.x * s + m[0], p.y * s + m[1])

    result_mod = [
        PoseResult(BodyResult([mul(keypoint, scale, move) for keypoint in body],
                               detection_results[0].body.total_score,
                               detection_results[0].body.total_parts),
                   [mul(keypoint, scale, move) for keypoint in hand_l],
                   [mul(keypoint, scale, move) for keypoint in hand_r],
                   [mul(keypoint, scale, move) for keypoint in face],)
    ]

    openpose_paint(result_mod, save_path)

    return scale, min_x, min_y


def scale_down_character(image: Image, scale: float, min_x: float, min_y: float):
    H, W = 512, 512
    image = image.resize((round(W / scale), round(H / scale)))
    output = Image.new('RGB', (W, H))
    index_x = round((min_x - 0.1 / scale) * W)
    index_y = round((min_y - 0.1 / scale) * H)
    output.paste(image, (index_x, index_y))

    return output


def face_blend(image_src: Image, image_out: Image, preprocessor, std_factor: float = 1.5, size: int = 512):
    pixels_out = image_out.load()
    pixels_src = image_src.load()

    face_points = preprocessor.detect_poses(np.array(image_src), include_face=True, include_hand=False)[0].face
    if face_points is None:
        return image_out
    face_x = np.array([point.x for point in face_points])
    face_y = np.array([point.y for point in face_points])
    mean_x = np.mean(face_x)
    mean_y = np.mean(face_y)
    std_x = np.std(face_x) * std_factor
    std_y = np.std(face_y) * std_factor

    left, right, up, down \
        = (round(size * (mean_x - std_x)), round(size * (mean_x + std_x)),
           round(size * (mean_y - std_y)), round(size * (mean_y + std_y)))
    print(left, right, up, down)
    for i in range(left, right + 1):
        for j in range(up, down + 1):
            r1, g1, b1 = pixels_out[i, j]
            r2, g2, b2 = pixels_src[i, j]
            ratio = min((abs(i / size - mean_x) / std_x + abs(j / size - mean_y) / std_y) / 2, 1.0)
            ratio *= ratio
            pixels_out[i, j] = (round(r1 * ratio + r2 * (1 - ratio)),
                                round(g1 * ratio + g2 * (1 - ratio)),
                                round(b1 * ratio + b2 * (1 - ratio)))
    return image_out


def face_correction(input_image: Image, save_path: str, preprocessor, points_only: bool = False, bias: float = 0.001,
                    face_length: float = 0.5, face_width: float = 0.5, eyebrow_height: float = 0.5,
                    eye_height: float = 0.5, nose_size: float = 0.5, mouth_size: float = 0.5):
    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)

    input_image = HWC3(input_image)
    input_image = resize_image(input_image, 512)
    results = preprocessor.detect_poses(input_image, include_hand=True, include_face=True)

    # face scale
    face_origin = results[0].face
    if face_origin is None:
        print("Openpose Detection Failed .. ")
        return results, None

    face_x = [point.x for point in face_origin]
    scale_x = max(face_x) - min(face_x)
    face_y = [point.y for point in face_origin]
    scale_y = max(face_y) - min(face_y)
    scale_origin = max(scale_x, scale_y)

    body = results[0].body
    if body is None:
        print("Openpose Detection Failed .. ")
        return results, None

    nose = body.keypoints[0]
    if nose is None:
        print("Openpose Detection Failed .. ")
        return results, None

    print("nose: ", nose)
    # left and right could be None
    left_eye = body.keypoints[-4]  # left eye
    right_eye = body.keypoints[-3]  # right eye
    print("left_eye: ", left_eye, "right_eye: ", right_eye)

    if left_eye is None:
        left_eye = Keypoint(x=nose.x - bias, y=nose.y)
    if right_eye is None:
        right_eye = Keypoint(x=nose.x + bias, y=nose.y)

    left_ear = body.keypoints[-2]
    right_ear = body.keypoints[-1]
    print("left_ear: ", left_ear, "right_ear: ", right_ear)

    if left_ear is None:
        left_ear = Keypoint(x=nose.x - bias, y=nose.y)
    if right_ear is None:
        right_ear = Keypoint(x=nose.x + bias, y=nose.y)

    try:
        ear_height = (left_ear.y + right_ear.y) / 2
        angle_x = 180 * math.asin((ear_height - nose.y) / (0.5 * scale_origin)) / math.pi
        print(angle_x)

        l = nose.x - left_eye.x
        r = right_eye.x - nose.x
        print("l: ", l, ", r: ", r)
        if l < r:
            angle_y = 180 * math.acos(l / r) / math.pi
            angle_y = angle_y % 180 / 2
            angle_y = -angle_y
            print("acos(l / r): ", math.acos(l / r))
        else:
            angle_y = 180 * math.acos( r / l) / math.pi
            angle_y = angle_y % 180 / 2
            print("math.acos(r / l)", math.acos(r / l))

        # acos(1) = 0 or 180 so use % 180
        # use /2 to map 0-90 degree to 0-45 degree
        print(angle_y)

        angle_z = 180 * math.asin((right_eye.y - left_eye.y) / (0.5 * scale_origin)) / math.pi
        print(angle_z)
    except ValueError:
        print("Fail to convert face")
        return results, None

    xs, ys, zs = load_face_coordinates("config/face_points.csv")
    if face_length != 0.5:
        xs, ys = adjust_face_length(xs, ys, face_length)
    if face_width != 0.5:
        xs, ys = adjust_face_width(xs, ys, face_width)
    if eyebrow_height != 0.5:
        xs, ys = adjust_eyebrow_height(xs, ys, eyebrow_height)
    if eye_height != 0.5:
        xs, ys = adjust_eye_height(xs, ys, eye_height)
    if nose_size != 0.5:
        xs, ys = adjust_nose_size(xs, ys, nose_size)
    if mouth_size != 0.5:
        xs, ys = adjust_mouth_size(xs, ys, mouth_size)

    zs = [z * 0.15 for z in zs]
    points = []
    for i in range(len(xs)):
        points.append([xs[i], ys[i], zs[i]])

    points = np.array(points)

    center = np.array([0.5, 0.5, 0.5])

    # if angle_y > 45: remove all points that z < z_of_nose
    angles = [angle_x, angle_y, angle_z]

    rotated_points = rotate_points(points, center, angles)
    scale = ((rotated_points[:, 0].max() - rotated_points[:, 0].min())
             + (rotated_points[:, 1].max() - rotated_points[:, 1].min())) / 2
    rotated_points = rotated_points * scale_origin / scale
    rotated_points[:, 0] -= rotated_points[30, 0] - nose.x  # 30 is the index of face keypoint
    rotated_points[:, 1] -= rotated_points[30, 1] - nose.y

    if points_only:
        return results, rotated_points

    H = 512
    W = 512
    # canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # canvas = draw_facepose(canvas, rotated_points)
    canvas = draw_poses(results, H, W, draw_body=True, draw_hand=False, draw_face=False)
    canvas = draw_facepose(canvas, rotated_points)

    detected_map = canvas
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    detected_map = Image.fromarray(detected_map)
    detected_map.save(save_path)
    return results, rotated_points


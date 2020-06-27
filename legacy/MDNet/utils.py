import numpy as np
import cv2


class Options:
    use_gpu = True
    model_path = "bin/mdnet_vot-otb.pth"
    img_size = 107
    padding = 16
    batch_pos = 32
    batch_neg = 96
    batch_neg_cand = 1024
    batch_test = 256
    n_samples = 256
    trans = 0.6
    scale = 1.05
    trans_limit = 1.5
    trans_pos = 0.1
    scale_pos = 1.3
    trans_neg_init = 1
    scale_neg_init = 1.6
    trans_neg = 2
    scale_neg = 1.3
    n_bbreg = 1000
    overlap_bbreg = [0.6, 1]
    trans_bbreg = 0.3
    scale_bbreg = 1.6
    aspect_bbreg = 1.1
    lr_init = 0.0005
    maxiter_init = 50
    n_pos_init = 500
    n_neg_init = 5000
    overlap_pos_init = [0.7, 1]
    overlap_neg_init = [0, 0.5]
    lr_update = 0.001
    maxiter_update = 15
    n_pos_update = 50
    n_neg_update = 200
    overlap_pos_update = [0.7, 1]
    overlap_neg_update = [0, 0.3]
    long_interval = 10
    n_frames_long = 100
    n_frames_short = 30
    grad_clip = 10
    lr_mult = {'fc6': 10}
    ft_layers = ['fc']


def overlap_ratio(rect1, rect2):
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image2(img, bbox, img_size=107, padding=16):
    x, y, w, h = np.array(bbox, dtype='float32')

    cx, cy = x + w/2, y + h/2

    if padding > 0:
        w += 2 * padding * w/img_size
        h += 2 * padding * h/img_size

    # List of transformation matrices
    matrices = []

    # Translation matrix to move patch center to origin
    translation_matrix = np.asarray([[1, 0, -cx],
                                     [0, 1, -cy],
                                     [0, 0, 1]], dtype=np.float32)
    matrices.append(translation_matrix)

    # Scaling matrix according to image size
    scaling_matrix = np.asarray([[img_size / w, 0, 0],
                                 [0, img_size / h, 0],
                                 [0, 0, 1]], dtype=np.float32)
    matrices.append(scaling_matrix)

    # Translation matrix to move patch center from origin
    revert_t_matrix = np.asarray([[1, 0, img_size / 2],
                                  [0, 1, img_size / 2],
                                  [0, 0, 1]], dtype=np.float32)
    matrices.append(revert_t_matrix)

    # Aggregate all transformation matrices
    matrix = np.eye(3)
    for m_ in matrices:
        matrix = np.matmul(m_, matrix)

    # Warp image, padded value is set to 128
    patch = cv2.warpPerspective(img,
                                matrix,
                                (img_size, img_size),
                                borderValue=128)
    return patch

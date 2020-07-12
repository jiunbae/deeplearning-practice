from typing import Tuple, Union, Any
import types

import cv2
import numpy as np
import torch
from torchvision import transforms


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            try:
                img, boxes, labels = t(img, boxes, labels)
            except TypeError:
                img = t(img)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return (image.astype(np.float32) - self.mean) / self.std, boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape

        if boxes is not None:
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 aspect: bool = False, box: bool = False):
        self.size = np.array(size)
        self.aspect = aspect
        self.box = box

    def __call__(self, image, boxes=None, labels=None):
        image_size = np.array(image.shape[:2])
        if not np.all(self.size == image_size):
            if self.aspect:
                scale = max(image_size / self.size[::-1])
                nh, nw = (image_size / scale).astype(np.int)
                dx, dy = (self.size - (nw, nh)) / 2
                ratio = image_size / (nh, nw)

                canvas = np.zeros((*self.size[::-1], 3), dtype=np.float32)
                canvas[int(dy):int(dy + nh), int(dx):int(dx + nw)] = cv2.resize(image, (nw, nh))
                image = canvas

            else:
                ratio = image_size / self.size
                image = cv2.resize(image, tuple(self.size))

            if self.box:
                boxes /= np.tile(ratio, 2)

                if self.aspect:
                    boxes[:, ::2] += dx
                    boxes[:, 1::2] += dy

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.size and overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __init__(self, vertical: bool = False, horizontal: bool = True):
        self.vertical = vertical
        self.horizontal = horizontal

    def __call__(self, image, boxes, classes):
        height, width, *_ = image.shape

        boxes = boxes.copy()
        if self.vertical and np.random.randint(2):
            image = cv2.flip(image, 0)
            boxes[:, 1::2] = height - boxes[:, 3::-2]

        if self.horizontal and np.random.randint(2):
            image = cv2.flip(image, 1)
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class Augmentation:
    def __init__(self):
        self.training = True
        self.augment = {
            'train': lambda *_: None,
            'test': lambda *_: None,
        }

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, img: Union[np.ndarray, torch.Tensor], boxes=None, labels=None) \
            -> Tuple[Union[np.ndarray, torch.Tensor], Any, Any]:

        if isinstance(self.augment, dict):
            augment = self.augment.get('train' if self.training else 'test')
        elif isinstance(self.augment, Compose):
            augment = self.augment

        return augment(img, boxes, labels)


class Amano(Augmentation):
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 mean: Tuple[float, float, float] = (111, 113, 110),
                 horizontal: bool = True, vertical: bool = True,
                 **kwargs):
        self.mean = mean
        self.size = size
        self.augment = {
            'train': Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                RandomMirror(horizontal=horizontal, vertical=vertical),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean)
            ]), 'test': Compose([
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean)
            ])
        }


class Detection(Augmentation):
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 mean: Tuple[float, float, float] = (97.06, 97.53, 95.62),
                 horizontal: bool = True, vertical: bool = True,
                 **kwargs):
        self.mean = mean
        self.size = size
        self.augment = {
            'train': Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                RandomMirror(horizontal=horizontal, vertical=vertical),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
            ]), 'test': Compose([
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
            ])
        }


class EfficientDet(Augmentation):
    def __init__(self, size: Tuple[int, int] = (512, 512),
                 mean: Tuple[float, float, float] = (.485, .456, .406),
                 std: Tuple[float, float, float] = (.229, .224, .225),
                 horizontal: bool = True, vertical: bool = True,
                 **kwargs):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = {
            'train': Compose([
                RandomMirror(horizontal=horizontal, vertical=vertical),
                Resize(self.size, aspect=True, box=True),
                Normalize(self.mean, self.std),
            ]), 'test': Compose([
                Resize(self.size, aspect=True, box=True),
                Normalize(self.mean, self.std),
            ])
        }


class COCO(Augmentation):
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 mean: Tuple[float, float, float] = (123, 117, 104), std: float = 1.,
                 **kwargs):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
        ])


class VOC(Augmentation):
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 mean: Tuple[float, float, float] = (123, 117, 104), std: float = 1.,
                 **kwargs):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # ConvertFromInts(),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop(),
            # RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
        ])

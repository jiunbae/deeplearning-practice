from typing import Tuple, List, Iterable
from math import sqrt
from itertools import product
from collections import namedtuple

import numpy as np
import torch

PriorSpec = namedtuple('Spec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])


class PriorBox(object):
    SPEC = PriorSpec

    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self, size: Tuple[int, int] = (300, 300),
                 variance: List[int] = None, aspect_ratios: List[List[int]] = None,
                 steps: List[int] = None, feature_map: List[int] = None,
                 min_sizes: List[int] = None, max_sizes: List[int] = None,
                 clip: bool = True, **kwargs):
        super(PriorBox, self).__init__()

        self.size = np.array(size)
        self.variance = variance or [.1, .2]

        self.config = [
            PriorSpec(feature_map, step, (min_size, max_size), aspect_ratio)
            for aspect_ratio, step, feature_map, min_size, max_size in zip(
                aspect_ratios or [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                steps or [8, 16, 32, 64, 100, 300],
                feature_map or [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]],
                min_sizes or [30, 60, 111, 162, 213, 264],
                max_sizes or [60, 111, 162, 213, 264, 315],
            )
        ]

        self.num_priors = len(self.config)
        self.clip = clip

        if any(filter(lambda x: x <= 0, self.variance)):
            raise ValueError('Variances must be greater than 0')

    def forward(self):
        priors = []

        for spec in self.config:
            scale = self.size / spec.shrinkage
            box_min, box_max = spec.box_sizes

            feat_size = spec.feature_map_size
            feat_size = reversed(feat_size) if isinstance(feat_size, Iterable) else (feat_size, feat_size)

            for j, i in product(*map(range, feat_size)):
                x_center, y_center = (i + .5, j + .5) / scale

                # small sized square box
                w, h = box_min / self.size
                priors.append([x_center, y_center, w, h])

                # big sized square box
                w, h = sqrt(box_max * box_min) / self.size
                priors.append([x_center, y_center, w, h])

                # change w, h ratio of the small sized box
                w, h = box_min / self.size
                for ratio in map(sqrt, spec.aspect_ratios):
                    priors.append([x_center, y_center, w * ratio, h / ratio])
                    priors.append([x_center, y_center, w / ratio, h * ratio])

        priors = torch.tensor(priors)

        if self.clip:
            priors = torch.clamp(priors, 0., 1.)

        return priors

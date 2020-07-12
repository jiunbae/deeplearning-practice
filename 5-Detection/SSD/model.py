import re
from typing import List, Iterable, Tuple, Union
from functools import reduce
from itertools import chain

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .mobilenet import MobileNetV1
from .loss import Loss
from .priorbox import PriorBox
from .layers import GraphPath, Warping
from .utils.config import Config

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Custom SSD backbone requires below things
        - backbone: return feature layers
        - extra: return extra layers
        - head: return location, confidence layers as tuple
        - APPENDIX: list of extract information (index, preprocess, name)
            e.g. [(23, nn.BatchNorm2d(512), 'L2Norm'), (35, None, None)]
    """
    LOSS = Loss
    BACKBONE = None
    APPENDIX = None
    OPTIMIZER = optim.SGD, {'lr': .0001, "momentum": .9, "weight_decay": 5e-4}
    SCHEDULER = lambda _: None, {}
    batch_size = 1

    @classmethod
    def new(cls, num_classes: int, batch_size: int,
            config=None, **kwargs):
        assert cls is not SSD, "Create new model instance by subclass caller"

        backbone = cls.backbone()
        extras = list(cls.extra())
        loc, conf = cls.head(backbone, extras, num_classes)
        appendix = cls.APPENDIX
        config = config or Config(None, 'SSD', cls)

        return cls(num_classes, batch_size,
                   backbone, extras, loc, conf, appendix,
                   config, **kwargs)

    def __init__(self, num_classes: int, batch_size: int,
                 backbone, extras, loc, conf, appendix,
                 config=None,
                 warping: bool = False, warping_mode: str = 'sum',
                 **kwargs):
        """

        :param num_classes:
        :param batch_size:
        :param backbone:
        :param extras:
        :param loc:
        :param conf:
        :param appendix:
        :param config:
        :param warping: trigger warping layers, one of 'all' or 'head'
        :param warping_mode: one of 'sum', 'average' or 'concat'
        :param kwargs:
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.config = config

        self.features = backbone
        self.appendix = appendix
        self.priors = PriorBox(**self.config.dump).forward()
        self.extras, self.loc, self.conf = map(nn.ModuleList, (extras, loc, conf))

        for _, layer, name in self.appendix:
            if isinstance(layer, nn.Module):
                self.add_module(name, layer)

        self.warping = config.warping
        self.warping_mode = config.warping_mode

    def detect(self, locations: torch.Tensor, confidences: torch.Tensor, prior_boxes: torch.Tensor) \
            -> torch.Tensor:
        if self.training:
            raise RuntimeError('use detect after enable eval mode')

        with torch.no_grad():
            from .lib.box import decode
            from torchvision.ops import nms

            confidences = F.softmax(confidences, dim=-1)
            num_priors = prior_boxes.size(0)

            output = torch.zeros(self.batch_size, self.num_classes, self.config.nms_top_k, 5) \
                if self.config.nms else None
            confidences = confidences.view(self.batch_size, num_priors, self.num_classes).transpose(2, 1)

            # Decode predictions into bounding boxes.
            for batch_index, (location, confidence) in enumerate(zip(locations, confidences)):
                decoded_boxes = decode(location, prior_boxes, self.config.variance)
                conf_scores = confidence.clone()

                if self.config.nms:
                    for class_index in range(1, self.num_classes):
                        # idx of highest scoring and non-overlapping boxes per class
                        conf_mask = conf_scores[class_index].gt(self.config.conf_thresh)
                        scores = conf_scores[class_index][conf_mask]

                        if scores.size(0) == 0:
                            continue

                        loc_mask = conf_mask.unsqueeze(1).expand_as(decoded_boxes)
                        boxes = decoded_boxes[loc_mask].view(-1, 4)

                        nms_index = nms(boxes, scores, self.config.nms_thresh)
                        (size, *_) = nms_index.size()
                        output[batch_index, class_index, :min(size, self.config.nms_top_k)] = torch.cat((
                            scores[nms_index[:self.config.nms_top_k]].unsqueeze(1),
                            boxes[nms_index[:self.config.nms_top_k]]
                        ), dim=1)

                # skip nms process for ignore torch script export error
                else:
                    if output is None:
                        output = torch.cat((
                            conf_scores.unsqueeze(-1),
                            decoded_boxes.repeat(self.num_classes, 1).view(-1, *decoded_boxes.shape),
                        ), dim=-1).unsqueeze(0)

                    else:
                        output = torch.cat((
                            output,
                            torch.cat((
                                conf_scores.unsqueeze(-1),
                                decoded_boxes.repeat(self.num_classes, 1).view(-1, *decoded_boxes.shape),
                            ), dim=-1).unsqueeze(0)
                        ))

            flt = output.contiguous().view(self.batch_size, -1, 5)
            _, idx = flt[:, :, 0].sort(1, descending=True)
            _, rank = idx.sort(1)
            flt[(rank < self.config.nms_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

            return output

    def forward(self, x: torch.Tensor) \
            -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch, topk, 7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors*4]
                    3: priorbox layers, Shape: [2, num_priors*4]
        """
        def _forward(tensor: torch.Tensor, module: nn.Module) \
                -> torch.Tensor:
            return module.forward(tensor)

        start, sources = 0, []

        # forward layers for extract sources
        for index, layer, *_ in self.appendix:
            x = reduce(_forward, [x, *self.features[start:index]])

            if isinstance(layer, GraphPath):
                x, y = layer(x, self.features[index])
                index += 1

            elif layer is not None:
                y = layer(x)

            else:
                y = x

            sources.append(y)
            start = index

        # forward remain parts
        x = reduce(_forward, [x, *self.features[start:]])

        for i, layer in enumerate(self.extras):
            x = _forward(x, layer)
            sources.append(x)

        def refine(source: torch.Tensor) \
                -> torch.Tensor:
            return source.permute(0, 2, 3, 1).contiguous()

        def reshape(tensor: torch.Tensor) \
                -> torch.Tensor:
            return torch.cat(tuple(map(lambda t: t.view(t.size(0), -1), tensor)), 1)

        locations, confidences = map(reshape, zip(*[(refine(loc(source)), refine(conf(source)))
                                                    for source, loc, conf in zip(sources, self.loc, self.conf)]))

        locations = locations.view(self.batch_size, -1, 4)
        confidences = confidences.view(self.batch_size, -1, self.num_classes)

        output = (locations, confidences, self.priors.to(x.device))

        if not self.training:
            output = self.detect(*output).to(x.device)

        return output

    @staticmethod
    def initializer(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                m.bias.data.zero_()

    def load(self, state_dict: dict = None):
        try:
            self.load_state_dict(state_dict)

        # if state dict is only vgg features
        except RuntimeError:
            try:
                self.features.load_state_dict(state_dict)

                self.extras.apply(self.initializer)
                self.loc.apply(self.initializer)
                self.conf.apply(self.initializer)

            # if state dict is legacy pre-trained features
            except RuntimeError:
                def refine(text, replace_map_, pattern_):
                    return pattern_.sub(lambda m: next(k for k, v in replace_map_.items() if m.group(0) in v), text)

                remove_prefix = ['source_layer_add_ons']
                replace_map = {
                    # https://github.com/qfgaohao/pytorch-ssd weights
                    'features': ['vgg', 'base_net'],
                    'loc': ['regression_headers'], 'conf': ['classification_headers'],
                    'extras.0.0': ['extras.0'], 'extras.0.2': ['extras.1'],
                    'extras.1.0': ['extras.2'], 'extras.1.2': ['extras.3'],
                    'extras.2.0': ['extras.4'], 'extras.2.2': ['extras.5'],
                    'extras.3.0': ['extras.6'], 'extras.3.2': ['extras.7'],
                }
                pattern = re.compile('|'.join(chain(*replace_map.values())))

                self.load_state_dict(state_dict.__class__({
                    refine(key, replace_map, pattern): value for key, value in state_dict.items()
                    if not any(map(key.startswith, remove_prefix))
                }), strict=False)

        except AttributeError:
            self.extras.apply(self.initializer)
            self.loc.apply(self.initializer)
            self.conf.apply(self.initializer)

    @classmethod
    def backbone(cls, *args, **kwargs):
        method, arguments = cls.BACKBONE

        return method(*args, **(kwargs.update(arguments) or kwargs)).features

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:
        raise NotImplementedError()

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        raise NotImplementedError()


class VGG16(SSD):
    BACKBONE = models.vgg16, {'pretrained': True}
    # SCHEDULER = schedulers.MultiStepLR, {'milestones': (80, 100), 'gamma': .1}
    APPENDIX = [(23, nn.BatchNorm2d(512), 'L2Norm'), (35, None, None)]
    EXTRAS = [(256, 512, 1), (128, 256, 1), (128, 256, 0), (128, 256, 0)]
    BOXES = [4, 6, 6, 6, 4, 4]

    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    feature_map = (38, 19, 10, 5, 3, 1)
    steps = (8, 16, 32, 64, 100, 300)
    sizes = ((30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315))

    @classmethod
    def backbone(cls, *args, **kwargs):
        method, arguments = cls.BACKBONE
        backbone = method(*args, **(kwargs.update(arguments) or kwargs)).features[:-1]
        backbone[16].ceil_mode = True

        for i, layer in enumerate([
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
        ], 30):
            backbone.add_module(str(i), layer)

        return backbone

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:

        for mid_channels, out_channels, option in cls.EXTRAS:
            yield nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3,
                          stride=1 + option, padding=option),
                nn.ReLU(),
            )
            in_channels = out_channels

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        def gen(count_feature):
            count, feature = count_feature
            return nn.Conv2d(feature, count * 4, kernel_size=3, padding=1), \
                nn.Conv2d(feature, count * num_classes, kernel_size=3, padding=1)

        return tuple(zip(*map(gen, zip(cls.BOXES, chain(
            map(lambda layer: layer.out_channels, map(lambda index: backbone[index[0] - 2], cls.APPENDIX)),
            map(lambda module: module[2].out_channels, extras)),
        ))))


class MOBILENET1(SSD):
    BACKBONE = MobileNetV1, {}
    # SCHEDULER = schedulers.CosineAnnealingLR, {'T_max': 120}
    APPENDIX = [(12, None, None), (14, None, None)]
    EXTRAS = [(256, 512, 1), (128, 256, 1), (128, 256, 1), (128, 256, 1)]

    aspect_ratios = ((2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3))
    feature_map = (19, 10, 5, 3, 2, 1)
    steps = (16, 32, 64, 100, 150, 300)
    sizes = ((60, 105), (105, 150), (150, 195), (195, 240), (240, 285), (285, 330))

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:

        for mid_channels, out_channels, option in cls.EXTRAS:
            yield nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3,
                          stride=1 + option, padding=option),
                nn.ReLU(),
            )
            in_channels = out_channels

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:

        regression_headers = [
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        ]

        classification_headers = [
            nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        ]

        return regression_headers, classification_headers


class MOBILENET1_LITE(SSD):
    BACKBONE = MobileNetV1, {}
    # SCHEDULER = schedulers.CosineAnnealingLR, {'T_max': 120}
    APPENDIX = [(12, None, None), (14, None, None)]
    EXTRAS = [(256, 512, 1), (128, 256, 1), (128, 256, 1), (128, 256, 1)]

    aspect_ratios = ((2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3))
    feature_map = (19, 10, 5, 3, 2, 1)
    steps = (16, 32, 64, 100, 150, 300)
    sizes = ((60, 105), (105, 150), (150, 195), (195, 240), (240, 285), (285, 330))

    @staticmethod
    def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      groups=in_channels, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    @classmethod
    def extra(cls, in_channels: int = 1024) \
            -> Iterable[nn.Module]:

        for mid_channels, out_channels, option in cls.EXTRAS:
            yield nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
                nn.ReLU(),
                cls.SeperableConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1 + option, padding=option),
            )
            in_channels = out_channels

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:

        regression_headers = [
            cls.SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),
        ]

        classification_headers = [
            cls.SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1),
        ]

        return regression_headers, classification_headers


class MOBILENET2_LITE(SSD):
    BACKBONE = models.mobilenet_v2, {'pretrained': True}
    # SCHEDULER = schedulers.CosineAnnealingLR, {'T_max': 120}
    APPENDIX = [(14, GraphPath('conv', 1), 'GraphPath'), (19, None, None)]
    EXTRAS = [(512, .2), (256, .25), (256, .5), (64, .25)]

    aspect_ratios = ((2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3))
    feature_map = (19, 10, 5, 3, 2, 1)
    steps = (16, 32, 64, 100, 150, 300)
    sizes = ((60, 105), (105, 150), (150, 195), (195, 240), (240, 285), (285, 330))

    @staticmethod
    def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                      groups=in_channels, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    @classmethod
    def extra(cls, in_channels: int = 1280) \
            -> Iterable[nn.Module]:

        for feature, ratio in cls.EXTRAS:
            yield models.mobilenet.InvertedResidual(in_channels, feature, stride=2, expand_ratio=ratio)
            in_channels = feature

    @classmethod
    def head(cls, backbone: nn.Module, extras: List[nn.Module], num_classes: int, width_mult: float = 1.0) \
            -> Tuple[Iterable[nn.Module], Iterable[nn.Module]]:
        in_channels = round(576 * width_mult)

        regression_headers = [
            cls.SeperableConv2d(in_channels, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            cls.SeperableConv2d(1280, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            cls.SeperableConv2d(512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            cls.SeperableConv2d(256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            cls.SeperableConv2d(256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ]

        classification_headers = [
            cls.SeperableConv2d(in_channels, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            cls.SeperableConv2d(256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ]

        return regression_headers, classification_headers

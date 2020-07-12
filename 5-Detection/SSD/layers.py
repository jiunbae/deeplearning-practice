from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GraphPath(nn.Module):

    def __init__(self, name, index):
        super(GraphPath, self).__init__()
        self.name = name
        self.index = index

    def forward(self, x: torch.Tensor, layer: nn.Module):
        sub = getattr(layer, self.name)

        for layer in sub[:self.index]:
            x = layer(x)

        y = x

        for layer in sub[self.index:]:
            x = layer(x)

        return x, y


class PositionConv2d(nn.Module):
    def __init__(self, n_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, padding: int = 0, stride: int = 1):
        super(PositionConv2d, self).__init__()

        self.n_channels = n_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.Tensor(
            self.out_channels, self.n_channels, kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.Tensor(
            self.out_channels))

    def forward(self, inputs: torch.Tensor):
        dtype, device = inputs.dtype, inputs.device
        b, c, w, h = inputs.shape

        width, height = (
            np.array((w, h)) + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1

        windows = F.unfold(inputs,
                           kernel_size=(self.kernel_size, self.kernel_size), padding=(self.padding, self.padding),
                           dilation=(self.dilation, self.dilation), stride=(self.stride, self.stride)) \
            .transpose(1, 2).contiguous().view(-1, c, self.kernel_size * self.kernel_size).transpose(0, 1)

        result = torch.zeros((b * self.out_channels, width, height), dtype=dtype, device=device)

        for out_index in range(self.out_channels):
            for window, weight in zip(windows, self.weight[out_index]):
                temp = torch.matmul(window, weight).view(-1, width, height)
                result[out_index*temp.shape[0]:(out_index+1)*temp.shape[0]] += temp

            result[out_index*b:(out_index+1)*b] += self.bias[out_index]

        return result.view(b, self.out_channels, width, height)


class Warping(Function):
    """Feature warping layer

    Support warping mode follows:
    - replace:  Return feature map after warping
    - fit:      Average of warped feature map and input feature map of fitting window
                (inside square of circular feature map)
    - sum:      Sum of warped feature map and input feature map
    - average:  Average of warped feature map and input feature map

    Apply warping layers
    - head: Apply head of features
    - all:  Apply all feature maps

    """
    PADDING = 480, 0
    SHAPE = 2880, 2880
    CALIBRATION = {
        'f': [998.4, 998.4],
        'c': [1997, 1473],
        'k': [0.0711, -0.0715, 0, 0, 0],
    }

    @classmethod
    def forward(cls, inputs: torch.Tensor, mode: str = '', grid: torch.Tensor = None) \
            -> torch.Tensor:
        size = sum(inputs.shape[2:]) / 2

        if size == 1:
            return inputs

        if grid is None:
            grid = torch.from_numpy(np.expand_dims(cls.grid(step=(20/size)), 0)).to(inputs.device)

        shape = grid.shape
        grid = grid.view(1, -1).repeat(1, inputs.shape[0]).view(-1, *shape[1:])

        output = F.grid_sample(inputs, grid)

        if mode == 'replace':
            pass

        elif mode == 'fit':
            size = np.array(shape[1:-1])
            scale = 2 ** -.5

            x, y = ((1 - scale) / 2 * size).astype(np.int)
            w, h = (size * scale).astype(np.int)

            resized = F.interpolate(output, size=(w, h))

            output = inputs.clone()
            output[:, :, x:x + w, y:y + h] = (inputs[:, :, x:x + w, y:y + h] + resized) / 2

        elif mode == 'sum':
            output += inputs

        elif mode == 'average':
            output += inputs
            output /= 2

        elif mode == 'concat':
            output = torch.cat((output, inputs), -1)

        else:
            raise NotImplementedError(f'Warping {mode} is not implemented!')

        return output

    @classmethod
    def grid(cls, wide: int = 10, step: float = 1.) \
            -> np.ndarray:
        arange = np.arange(-wide, wide, step)
        grid = np.array(np.meshgrid(arange, arange), dtype=np.float32).transpose(1, 2, 0)
        shape = grid.shape
        grid = np.apply_along_axis(lambda x: cls.ray2pix([*x, 3]), 1, grid.reshape(-1, 2))

        grid[:, 0] -= cls.PADDING[0]
        grid[:, 1] -= cls.PADDING[1]

        grid[:, 0] /= cls.SHAPE[0]
        grid[:, 1] /= cls.SHAPE[1]

        return grid.reshape(shape) * 2 - 1

    @classmethod
    def ray2pix(cls, ray: Union[List, np.ndarray]) \
            -> np.ndarray:
        ray = np.array(ray)

        if np.all(ray[:2] == 0):
            return np.array(cls.CALIBRATION['c'])

        nr = ray / np.square(ray).sum()
        d = np.sqrt((nr[:2] * nr[:2]).sum())
        th = np.arctan2(d, nr[-1])
        th = th * (1 + th * (cls.CALIBRATION['k'][0] + th * cls.CALIBRATION['k'][1]))
        q = nr[:2] * (th / d)
        im = np.asarray([[cls.CALIBRATION['f'][0], 0, cls.CALIBRATION['c'][0]],
                         [0, cls.CALIBRATION['f'][1], cls.CALIBRATION['c'][1]],
                         [0, 0, 1]], dtype=np.float32)
        return (im @ np.asarray([[*q, 1]], dtype=np.float32).T).T.squeeze()[:2]

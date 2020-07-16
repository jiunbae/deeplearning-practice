import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

from ..utils import *


class SampleGenerator(object):
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size) # (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):
        def _gen_samples(bb, n):
            bb = np.array(bb, dtype='float32')

            sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
            samples = np.tile(sample[None, :], (n, 1))

            if self.aspect is not None:
                ratio = np.random.rand(n, 2) * 2 - 1
                samples[:, 2:] *= self.aspect ** ratio

            if self.type == 'gaussian':
                samples[:, :2] += self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
                samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

            elif self.type == 'uniform':
                samples[:, :2] += self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
                samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

            elif self.type == 'whole':
                m = int(2 * np.sqrt(n))
                xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
                xy = np.random.permutation(xy)[:n]
                samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
                samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

            samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
            if self.valid:
                samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
            else:
                samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

            samples[:, :2] -= samples[:, 2:] / 2

            return samples

        if overlap_range is None and scale_range is None:
            return _gen_samples(bbox, n)

        else:
            samples = None
            remain = n
            factor = 2
            while remain > 0 and factor < 16:
                samples_ = _gen_samples(bbox, remain * factor)

                idx = np.ones(len(samples_), dtype=bool)
                if overlap_range is not None:
                    r = overlap_ratio(samples_, bbox)
                    idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                if scale_range is not None:
                    s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                samples_ = samples_[idx, :]
                samples_ = samples_[:min(remain, len(samples_))]
                if samples is None:
                    samples = samples_
                else:
                    samples = np.concatenate([samples, samples_])
                remain = n - len(samples)
                factor = factor * 2

            return samples


class RegionExtractor(object):
    def __init__(self, image, samples, crop_size, padding, batch_size):
        self.image = np.asarray(image)
        self.samples = samples
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions


class RegionDataset(data.Dataset):
    def __init__(self, img_list, gt, opts):
        self.img_list = np.asarray(img_list)
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('uniform', image.size,
                opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size,
                opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions)
        neg_regions = torch.from_numpy(neg_regions)
        return pos_regions, neg_regions

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions

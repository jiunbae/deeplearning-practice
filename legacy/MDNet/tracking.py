import argparse
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import Options, overlap_ratio
from models.mdnet import MDNet, BCELoss
from models.extractor import SampleGenerator, RegionExtractor
from models.regressor import BBRegressor


def forward_samples(model, image, samples, opts, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts.img_size, opts.padding, opts.batch_test)

    for i, regions in enumerate(extractor):
        if opts.use_gpu:
            regions = regions.cuda()

        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)

        feats = torch.cat((feats, feat.detach().clone()), 0) if i else feat.detach().clone()
    return feats


def train(model, criterion, optimizer,
          pos_feats, neg_feats, maxiter, opts,
          in_layer='fc4'):
    model.train()

    batch_pos = opts.batch_pos
    batch_neg = opts.batch_neg
    batch_test = opts.batch_test
    batch_neg_cand = max(opts.batch_neg_cand, batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))

    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])

    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])

    pos_pointer = 0
    neg_pointer = 0

    for _ in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()

            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)

                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)

                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)

        model.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)

        optimizer.step()


def main(images, init_bbox, ground_truths, opts):
    device = ('cuda' if opts.use_gpu else 'cpu')

    model = MDNet(opts.model_path).to(device)

    criterion = BCELoss()

    # Set learnable parameters
    for k, p in model.params.items():
        p.requires_grad = any([k.startswith(l) for l in opts.ft_layers])

    # Set optimizer states
    def set_optimizer(lr_base, lr_mult, momentum=0.9, w_decay=0.0005):
        param_list = []

        for k, p in filter(lambda kp: kp[1].requires_grad, model.params.items()):
            lr = lr_base
            for l, m in lr_mult.items():
                if k.startswith(l):
                    lr = lr_base * m
            param_list.append({'params': [p], 'lr': lr})

        return optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)

    init_optimizer = set_optimizer(opts.lr_init, opts.lr_mult)
    update_optimizer = set_optimizer(opts.lr_update, opts.lr_mult)

    # Load first image
    image = Image.open(images[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts.trans_pos, opts.scale_pos)(
        init_bbox, opts.n_pos_init, opts.overlap_pos_init)

    neg_examples = np.concatenate([
        SampleGenerator('uniform', image.size, opts.trans_neg_init, opts.scale_neg_init)(
            init_bbox, int(opts.n_neg_init * 0.5), opts.overlap_neg_init),
        SampleGenerator('whole', image.size)(
            init_bbox, int(opts.n_neg_init * 0.5), opts.overlap_neg_init)])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples, opts)
    neg_feats = forward_samples(model, image, neg_examples, opts)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts.maxiter_init, opts)
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox Regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts.trans_bbreg, opts.scale_bbreg, opts.aspect_bbreg)\
        (init_bbox, opts.n_bbreg, opts.overlap_bbreg)

    bbreg_feats = forward_samples(model, image, bbreg_examples, opts)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, init_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts.trans, opts.scale)
    pos_generator = SampleGenerator('gaussian', image.size, opts.trans_pos, opts.scale_pos)
    neg_generator = SampleGenerator('uniform', image.size, opts.trans_neg, opts.scale_neg)

    # Init pos/neg features for update
    neg_examples = neg_generator(init_bbox, opts.n_neg_update, opts.overlap_neg_init)
    neg_feats = forward_samples(model, image, neg_examples, opts)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    # Main loop
    for i, image in enumerate(images[1:], 1):
        image = Image.open(image).convert('RGB')

        # Estimate target bbox
        samples = sample_generator(init_bbox, opts.n_samples)
        sample_scores = forward_samples(model, image, samples, opts, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        init_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            init_bbox = init_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        sample_generator.trans = opts.trans if success else min(sample_generator.trans * 1.1, opts.trans_limit)

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]

            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]

            bbreg_feats = forward_samples(model, image, bbreg_samples, opts)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)

        else:
            bbreg_bbox = init_bbox

        yield init_bbox, bbreg_bbox, overlap_ratio(ground_truths[i], bbreg_bbox)[0], target_score

        # Data collect
        if success:
            pos_examples = pos_generator(init_bbox, opts.n_pos_update, opts.overlap_pos_update)
            pos_feats = forward_samples(model, image, pos_examples, opts)
            pos_feats_all.append(pos_feats)

            if len(pos_feats_all) > opts.n_frames_long:
                del pos_feats_all[0]

            neg_examples = neg_generator(init_bbox, opts.n_neg_update, opts.overlap_neg_update)
            neg_feats = forward_samples(model, image, neg_examples, opts)
            neg_feats_all.append(neg_feats)

            if len(neg_feats_all) > opts.n_frames_short:
                del neg_feats_all[0]

        # Short term update
        # TODO: What if disable Short term upate?
        if not success:
            nframes = min(opts.n_frames_short, len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts.maxiter_update, opts)

        # Long term update
        # TODO: What if disable Long term update?
        elif i % opts.long_interval == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts.maxiter_update, opts)

        torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='./datasets/DragonBaby', help='input seq')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    options = Options()
    dataset = Path(args.seq)

    images = list(sorted(dataset.joinpath('img').glob('*.jpg')))
    ground_truths = pd.read_csv(str(dataset.joinpath('groundtruth_rect.txt')), header=None).values

    # Run tracker
    for i, (result, (x, y, w, h), overlap, score) in \
            enumerate(main(images, ground_truths[0], ground_truths, options), 1):
        image = np.asarray(Image.open(images[i]).convert('RGB'))

        print(i, result)

        gx, gy, gw, gh = ground_truths[i]
        cv2.rectangle(image, (int(gx), int(gy)), (int(gx+gw), int(gy+gh)), (0, 255, 0), 2)
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

        plt.imshow(image)
        plt.pause(.1)
        plt.draw()

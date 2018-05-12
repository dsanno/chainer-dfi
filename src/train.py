import argparse
import os
from PIL import Image
import numpy as np
import six

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import serializers

from net import VGG19
from lbfgs import LBFGS

input_image_size = (200, 200)

def parse_arg():
    parser = argparse.ArgumentParser('Deep Feature Interpolation')
    parser.add_argument('input_image', type=str, help='Image file path to be interpolated')
    parser.add_argument('output_image', type=str, help='Output image file path')
    parser.add_argument('source_list', type=str, help='Source image list file path')
    parser.add_argument('target_list', type=str, help='Target image list file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg19.model', help='Model file path')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='Mini batch size')
    parser.add_argument('--lr', '-l', type=float, default=1, help='Learning rate')
    parser.add_argument('--iter', '-i', type=int, default=1000)
    parser.add_argument('--clip-rect', '-c', type=str, default=None, help='Clipping rect for source/target images: (left, top, right, bottom)')
    parser.add_argument('--input-clip-rect', type=str, default=None, help='Clipping rect for input image: (left, top, right, bottom)')
    parser.add_argument('--max-image', type=int, default=2000, help='Maximum number of source/target images to be loaded')
    parser.add_argument('--near-image', type=int, default=100, help='Maximum number of source/target images for nearest neighbor')
    parser.add_argument('--tv-weight', type=float, default=100.0, help='Total variation loss weight')
    parser.add_argument('--single-weight-mode', type=float, help='When specified runs DFI only for supplied weight')
    return parser.parse_args()

def preprocess_image(image, image_size, clip_rect=None):
    if clip_rect is not None:
        image = image.crop(clip_rect)
    image = image.resize(image_size, Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32)
    return VGG19.preprocess(x, input_type='RGB')

def postprocess_image(original_image, diff):
    diff = diff.transpose((0, 2, 3, 1))
    diff = diff.reshape(diff.shape[1:])[:,:,::-1]
    diff = (diff + 128).clip(0, 255).astype(np.uint8)
    diff_image = Image.fromarray(diff).resize(original_image.size, Image.BILINEAR)
    image = np.asarray(original_image, dtype=np.int32) + np.asarray(diff_image, dtype=np.int32) - 128
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))

def list_dir_image(path, max_size):
    files = os.listdir(path)
    paths = []
    for file_name in files:
        name, ext = os.path.splitext(file_name)
        if not ext in ['.jpg', '.jpeg', '.png', '.gif']:
            continue
        paths.append(os.path.join(path, file_name))
        if len(paths) >= max_size:
            break
    return paths

def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def parse_numbers(rect_str):
    if not rect_str:
        return None
    return tuple(map(int, rect_str.split(',')))

def feature(net, x, layers=['3_1', '4_1', '5_1']):
    y = net(x)
    return [F.reshape(y[layer], (y[layer].shape[0], -1)) for layer in layers]

def rank_image(net, paths, image_size, image, top_num, clip_rect=None):
    xp = net.xp
    image_num = len(paths)
    diffs = []
    for path in paths:
        im = Image.open(path).convert('RGB')
        im = preprocess_image(Image.open(path), image_size, clip_rect)
        diffs.append(np.sum(np.square(image - im)))
    diffs = np.asarray(diffs, dtype=np.float32)
    rank = np.argsort(diffs)
    return [paths[r] for r in rank[:top_num]]

def mean_feature(net, paths, image_size, base_feature, top_num, batch_size, clip_rect=None):
    xp = net.xp
    image_num = len(paths)
    features = []
    for i in six.moves.range(0, image_num, batch_size):
        x = [preprocess_image(Image.open(path).convert('RGB'), image_size, clip_rect) for path in paths[i:i + batch_size]]
        x = xp.asarray(np.concatenate(x, axis=0))
        y = feature(net, x)
        features.append([cuda.to_cpu(layer.data) for layer in y])
    if image_num > top_num:
        last_features = np.concatenate([f[-1] for f in features], axis=0)
        last_features = last_features.reshape((last_features.shape[0], -1))
        base_feature = cuda.to_cpu(base_feature).reshape((1, -1,))
        diff = np.sum((last_features - base_feature) ** 2, axis=1)

        nearest_indices = np.argsort(diff)[:top_num]
        nearests = [np.concatenate(xs, axis=0)[nearest_indices] for xs in zip(*features)]
    else:
        nearests = [np.concatenate(xs, axis=0) for xs in zip(*features)]

    return [xp.asarray(np.mean(f, axis=0, keepdims=True)) for f in nearests]

def normalized_diff(s, t):
    xp = cuda.get_array_module(s)
    w = t - s
    norm = xp.asarray(np.linalg.norm(cuda.to_cpu(w), axis=1, keepdims=True))
    return w / norm

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.shape
    wh = xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32)
    ww = xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32)
    return (F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)) / np.prod(x.shape, dtype=np.float32)

def update(net, optimizer, link, target_layers, tv_weight=0.001):
    layers = feature(net, link.x)
    total_loss = 0
    losses = []
    for layer, target in zip(layers, target_layers):
        loss = F.mean_squared_error(layer, target)
        losses.append(float(loss.data))
        total_loss += loss
    tv_loss = tv_weight * total_variation(link.x)
    losses.append(float(tv_loss.data))
    total_loss += tv_loss
    link.cleargrads()
    total_loss.backward()
    optimizer.update()
    return losses

def adjust_color_distribution(x, mean, std):
    m = np.mean(x, axis=(2, 3), keepdims=True)
    s = np.std(x, axis=(2, 3), keepdims=True)
    return (x - m) / s * std + mean

def find_nearest(xs, t):
    return min(xs, key=lambda x: np.linalg.norm(x - t))

def train(args, image_path, source_image_paths, target_image_paths, input_clip_rect=None, clip_rect=None):
    iteration = args.iter
    batch_size = args.batch_size
    device_id = args.gpu
    lr = args.lr
    tv_weight = args.tv_weight
    near_image_num = args.near_image

    isSingleModeOn = False
    initialWeight = 0.1
    rangeUpperBound = 6

    if args.single_weight_mode is not None:
        print('Single weight mode is on, calculating for w = ' + args.single_weight_mode)
        isSingleModeOn = True
        initialWeight = args.single_weight_mode
        rangeUpperBound = 2

    make_dir(os.path.split(args.output_image)[0])
    net = VGG19()
    serializers.load_npz(args.model, net)
    if device_id >= 0:
        net.to_gpu(device_id)
    xp = net.xp

    original_image = Image.open(image_path).convert('RGB')
    if input_clip_rect is not None:
        original_image = original_image.crop(input_clip_rect)
    image = preprocess_image(original_image, input_image_size)
    image_mean = np.mean(image, axis=(2, 3), keepdims=True)
    image_std = np.std(image, axis=(2, 3), keepdims=True)
    x = xp.asarray(image)
    org_layers = feature(net, x)
    org_layers = [layer.data for layer in org_layers]
    org_layer_norms = [xp.asarray(np.linalg.norm(cuda.to_cpu(layer), axis=1, keepdims=True)) for layer in org_layers]

    print('Calculating source feature')
    if len(source_image_paths) > near_image_num:
        source_image_paths = rank_image(net, source_image_paths, input_image_size, image, near_image_num, clip_rect)
    source_feature = mean_feature(net, source_image_paths, input_image_size, org_layers[-1], near_image_num, batch_size, clip_rect)

    print('Calculating target feature')
    if len(target_image_paths) > near_image_num:
        target_image_paths = rank_image(net, target_image_paths, input_image_size, image, near_image_num, clip_rect)
    target_feature = mean_feature(net, target_image_paths, input_image_size, org_layers[-1], near_image_num, batch_size, clip_rect)

    attribute_vectors = [normalized_diff(s, t) for s, t in zip(source_feature, target_feature)]

    base, ext = os.path.splitext(args.output_image)
    residuals = []
    initial_x = xp.random.uniform(-10, 10, x.shape).astype(np.float32)
    print('Calculating residuals')
    link = chainer.Link(x=x.shape)
    if device_id >= 0:
        link.to_gpu(device_id)
    link.x.data[...] = initial_x
    optimizer = LBFGS(lr, stack_size=5)
    optimizer.setup(link)
    for j in six.moves.range(600):
        losses = update(net, optimizer, link, org_layers, tv_weight)
        if (j + 1) % 20 == 0:
            z = cuda.to_cpu(link.x.data)
            z = adjust_color_distribution(z, image_mean, image_std)
            residuals.append(z - image)
    for i in six.moves.range(1, rangeUpperBound):

        if isSingleModeOn:
            w = initialWeight
        else:
            w = i * 0.1

        print('Generating image for weight: {0:.2f}'.format(w))
        link = chainer.Link(x=x.shape)
        if device_id >= 0:
            link.to_gpu(device_id)
        link.x.data[...] = initial_x
        target_layers = [layer + w * n * a for layer, n, a in zip(org_layers, org_layer_norms,  attribute_vectors)]
        optimizer = LBFGS(lr, stack_size=5)
        optimizer.setup(link)
        for j in six.moves.range(iteration):
            losses = update(net, optimizer, link, target_layers, tv_weight)
            if (j + 1) % 100 == 0:
                print('iter {} done loss:'.format(j + 1))
                print(losses)
            if (j + 1) % 500 == 0:
                z = cuda.to_cpu(link.x.data)
                z = adjust_color_distribution(z, image_mean, image_std)
                z -= find_nearest(residuals, z - image)
                file_name = '{0}_{1:02d}_{2:04d}{3}'.format(base, i, j + 1, ext)
                postprocess_image(original_image, z - image).save(file_name)
        z = cuda.to_cpu(link.x.data)
        z = adjust_color_distribution(z, image_mean, image_std)
        z -= find_nearest(residuals, z - image)
        file_name = '{0}_{1:02d}{2}'.format(base, i, ext)
        postprocess_image(original_image, z - image).save(file_name)
        print('Completed')

def main():
    args = parse_arg()
    with open(args.source_list) as f:
        source_image_paths = [line.strip() for line in f]
    with open(args.target_list) as f:
        target_image_paths = [line.strip() for line in f]
    clip_rect = parse_numbers(args.clip_rect)
    if clip_rect is not None:
        if len(clip_rect) != 4:
            print('clip-rect ({}) is invalid'.format(args.clip_rect))
            exit()
        left, top, right, bottom = clip_rect
        if left >= right or top >= bottom:
            print('clip-rect ({}) is empty'.format(args.clip_rect))
            exit()
    input_clip_rect = parse_numbers(args.input_clip_rect)
    if input_clip_rect is not None and len(input_clip_rect) != 4:
        print('input-clip-rect {} is invalid'.format(args.input_clip_rect))
        left, top, right, bottom = input_clip_rect
        if left >= right or top >= bottom:
            print('clip-rect ({}) is empty'.format(args.input_clip_rect))
            exit()
    train(args, args.input_image, source_image_paths, target_image_paths, input_clip_rect, clip_rect)

if __name__ == '__main__':
    main()

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
    parser.add_argument('source_dir', type=str, help='Source image directory path')
    parser.add_argument('target_dir', type=str, help='Target image directory path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg19.model', help='Model file path')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='Mini batch size')
    parser.add_argument('--lr', '-l', type=float, default=1, help='Learning rate')
    parser.add_argument('--iter', '-i', type=int, default=1000)
    parser.add_argument('--max-image', type=int, default=100, help='Maximum number of source/target images')
    parser.add_argument('--tv-weight', type=float, default=0.001, help='Total variation loss weight')
    return parser.parse_args()

def load_image(path, image_size):
    image = Image.open(path).resize(image_size)
    x = np.asarray(image, dtype=np.float32)
    return VGG19.preprocess(x, input_type='RGB')

def save_image(path, x):
    x = VGG19.postprocess(x, output_type='RGB')
    x = x.clip(0, 255).astype(np.uint8)
    Image.fromarray(x).save(path)

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

def feature(net, x, layers=['3_1', '4_1', '5_1']):
    y = net(x)
    y = [y[layer] for layer in layers]
    return y

def mean_feature(net, paths, image_size, batch_size):
    xp = net.xp
    image_num = len(paths)
    features = []
    for i in six.moves.range(0, image_num, batch_size):
        x = [load_image(path, image_size) for path in paths[i:i + batch_size]]
        x = xp.asarray(np.concatenate(x, axis=0))
        y = feature(net, x)
        features.append([xp.sum(layer.data, axis=0, keepdims=True) for layer in y])
    return map(lambda xs: xp.sum(xp.concatenate(xs, axis=0), axis=0, keepdims=True) / image_num, zip(*features))

def feature_diff(s, t):
    xp = cuda.get_array_module(s)
    v = t - s
    print v[:,:10,:10,:10]
    print float(xp.sqrt(xp.square(v).sum())), float(xp.sqrt(xp.square(t).sum())), float(xp.sqrt(xp.square(s).sum()))
#    v / xp.sqrt(xp.square(v).sum())
    return v

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.shape
    wh = xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32)
    ww = xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32)
    return (F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)) / np.prod(x.shape, dtype=np.float32)

def update(net, optimizer, link, target_layers, tv_weight=0.001):
    layers = feature(net, link.x)
    loss = 0
    for layer, target in zip(layers, target_layers):
        loss += F.mean_squared_error(layer, target)
    tv_loss = tv_weight * total_variation(link.x)
    print float(loss.data), float(tv_loss.data)
    loss += tv_loss
    link.cleargrads()
    loss.backward()
    optimizer.update()
    return float(loss.data)

def optimize(net, link, target_layers, iteration):
    optimizer = LBFGS(size=10)
    optimizer.setup(link)
    for i in six.moves.range(iteration):
        update(net, optimizer, link, target_layers)
    return link.x.data

def main():
    args = parse_arg()
    iteration = args.iter
    batch_size = args.batch_size
    device_id = args.gpu
    lr = args.lr
    tv_weight = args.tv_weight
    net = VGG19()
    serializers.load_npz(args.model, net)
    if device_id >= 0:
        net.to_gpu(device_id)
    xp = net.xp

    source_image_files = list_dir_image(args.source_dir, args.max_image)
    source_feature = mean_feature(net, source_image_files, input_image_size, batch_size)
    target_image_files = list_dir_image(args.target_dir, args.max_image)
    target_feature = mean_feature(net, target_image_files, input_image_size, batch_size)
    attribute_vector = [feature_diff(s, t) for s, t in zip(source_feature, target_feature)]

    x = xp.asarray(load_image(args.input_image, input_image_size))
    layers = feature(net, x)
    layers = [layer.data for layer in layers]

    base, ext = os.path.splitext(args.output_image)
    for i in six.moves.range(0, 11):
        w = i * 1.0
        print('Generating image for weight: {0:.2f}'.format(w))
        link = chainer.Link(x=x.shape)
        if device_id >= 0:
            link.to_gpu(device_id)
#        link.x.data[...] = x
        link.x.data[...] = xp.random.uniform(-10, 10, x.shape).astype(np.float32)
        target_layers = [layer + w * a for layer, a in zip(layers, attribute_vector)]
        optimizer = LBFGS(lr, size=10)
#        optimizer = chainer.optimizers.Adam(lr)
        optimizer.setup(link)
        for j in six.moves.range(iteration):
            loss = update(net, optimizer, link, target_layers, tv_weight)
            if (j + 1) % 500 == 0:
                save_image('{0}_{1:02d}_{2:04d}{3}'.format(base, i, j + 1, ext), cuda.to_cpu(link.x.data))
        save_image('{0}_{1:02d}{2}'.format(base, i, ext), cuda.to_cpu(link.x.data))
        print('Completed')

if __name__ == '__main__':
    main()

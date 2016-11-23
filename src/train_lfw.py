import argparse
import numpy as np
import os

from train import train

def parse_arg():
    parser = argparse.ArgumentParser('Deep Feature Interpolation using LFW dataset')
    parser.add_argument('image_dir', type=str, help='Image directory path')
    parser.add_argument('attr_file', type=str, help='Attribute file path')
    parser.add_argument('input_name', type=str, help='Input person name')
    parser.add_argument('input_index', type=int, help='Input image index')
    parser.add_argument('output_image', type=str, help='Output image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index (negative value indicate CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg19.model', help='Model file path')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='Mini batch size')
    parser.add_argument('--lr', '-l', type=float, default=1, help='Learning rate')
    parser.add_argument('--iter', '-i', type=int, default=1000)
    parser.add_argument('--near-image', type=int, default=100, help='Maximum number of source/target images for nearest neighbor')
    parser.add_argument('--tv-weight', type=float, default=100.0, help='Total variation loss weight')
    return parser.parse_args()

def load_attribute_dataset(file_path):
    dtype = 'S50,i4' + ',f4' * 73
    data = np.genfromtxt(file_path, delimiter='\t', skip_header=2, dtype=dtype)
    data = zip(*data)
    names = data[0]
    numbers = data[1]
    attributes = np.asarray(data[2:], dtype=np.float32).T
    return names, numbers, attributes

def make_image_path(image_dir, name, num):
    name = name.replace(' ', '_')
    file_name = '{0}_{1:04d}.jpg'.format(name, num)
    return os.path.join(image_dir, name, file_name)

def image_paths(image_dir, attribute_dataset, indices):
    names, numbers, attributes = attribute_dataset
    return [make_image_path(image_dir, name, num) for name, num in zip(names, numbers)]

def nearest_attributes(attribute_dataset, attribute, attribute_id, positive, nearest_num):
    names, numbers, attributes = attribute_dataset
    indices = np.arange(len(names))
    if positive:
        selected = attributes[:,attribute_id] >= 0.5
    else:
        selected = attributes[:,attribute_id] < -0.5
    distance = np.linalg.norm(attributes[selected] - attribute, axis=1)
    tops = np.argsort(distance)[:nearest_num]
    return np.arange(attributes.shape[0])[selected][tops]

def find_attribute(attribute_dataset, name, index):
    names, numbers, attributes = attribute_dataset
    i = zip(names, numbers).index((name, index))
    return attributes[i]

def main():
    args = parse_arg()
    attribute_dataset = load_attribute_dataset(args.attr_file)
    attribute = find_attribute(attribute_dataset, args.input_name, args.input_index)
    # TODO args
    attribute_id = 17
    source_indices = nearest_attributes(attribute_dataset, attribute, attribute_id, False, args.near_image)
    target_indices = nearest_attributes(attribute_dataset, attribute, attribute_id, True, args.near_image)
    source_paths = image_paths(args.image_dir, attribute_dataset, source_indices)
    target_paths = image_paths(args.image_dir, attribute_dataset, target_indices)
    image_path = make_image_path(args.image_dir, args.input_name, args.input_index)
    train(args, image_path, source_paths, target_paths, (40, 20, 210, 190))

if __name__ == '__main__':
    main()

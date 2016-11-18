import argparse
import numpy as np
import os
import six
from PIL import Image

attributes_str = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'
attribute_names = attributes_str.lower().split()
attribute_ids = {k:i for i, k in enumerate(attribute_names)}
crop_area = (19, 39, 159, 179)

def parse_arg():
    parser = argparse.ArgumentParser('Extract face images')
    parser.add_argument('image_dir', type=str, help='Image directory')
    parser.add_argument('attr_file', type=str, help='Attribute file path')
    parser.add_argument('out_dir1', type=str, help='Output directory for images with specific feature')
    parser.add_argument('out_dir2', type=str, help='Output directory for images without specific feature')
    parser.add_argument('feature', type=str, choices=attribute_names, help='Attribute to separate output images')
    parser.add_argument('include', type=str, help='Attribute(s) output images should have')
    parser.add_argument('--exclude', '-e', type=str, default='', help='Attribute(s) output images should not have')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run')
    return parser.parse_args()

def extract(input_file, output_dir):
    _, file_name = os.path.split(input_file)
    image = Image.open(input_file).crop(crop_area)
    image.save(os.path.join(output_dir, file_name))

def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def main():
    args = parse_arg()
    attribute_size = len(attribute_names)
    attribute_flag = np.zeros(attribute_size, dtype=np.int32)
    feature_id = attribute_ids[args.feature]
    include_attributes = args.include.split(',')
    if args.exclude:
        exclude_attributes = args.exclude.split(',')
    else:
        exclude_attributes = []
    for attr in include_attributes:
        if not attr in attribute_names:
            print('Warning {} is invalid attribute'.format(attr))
            continue
        attribute_flag[attribute_ids[attr]] = 1
    for attr in exclude_attributes:
        if not attr in attribute_names:
            print('Warning {} is invalid attribute'.format(attr))
            continue
        attribute_flag[attribute_ids[attr]] = -1
    make_dir(args.out_dir1)
    make_dir(args.out_dir2)
    cols = six.moves.range(1, attribute_size + 1)
    attributes = np.loadtxt(args.attr_file, skiprows=2, usecols=cols, dtype=np.int32)
    positive_num = 0
    negative_num = 0
    for i, attr in enumerate(attributes):
        if not np.all(attr * attribute_flag >= 0):
            continue
        if attr[feature_id] >= 0:
            output_dir = args.out_dir1
            positive_num += 1
        else:
            output_dir = args.out_dir2
            negative_num += 1
        if args.dry_run:
            continue
        file_path = os.path.join(args.image_dir, '{0:06d}.jpg'.format(i + 1))
        extract(file_path, output_dir)
    print('{} images with feature'.format(positive_num))
    print('{} images without feature'.format(negative_num))

if __name__ == '__main__':
    main()

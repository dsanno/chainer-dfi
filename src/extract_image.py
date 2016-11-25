import argparse
import numpy as np
import os
import six
from PIL import Image

attributes_str = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'
attribute_names = attributes_str.lower().split()
attribute_ids = {k:i for i, k in enumerate(attribute_names)}

def parse_arg():
    parser = argparse.ArgumentParser('Extract face images that have specific features from CelebA dataset')
    parser.add_argument('image_dir', type=str, help='Image directory')
    parser.add_argument('attr_file', type=str, help='Attribute file path')
    parser.add_argument('source_list', type=str, help='Output directory for images without specific feature')
    parser.add_argument('target_list', type=str, help='Output directory for images with specific feature')
    parser.add_argument('feature', type=str, help='Attribute(s) to separate output images')
    parser.add_argument('include', type=str, help='Attribute(s) output images should have')
    parser.add_argument('--exclude', '-e', type=str, default='', help='Attribute(s) output images should not have')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Dry run')
    return parser.parse_args()

def main():
    args = parse_arg()
    attribute_size = len(attribute_names)
    attribute_flag = np.zeros(attribute_size, dtype=np.int32)
    features = args.feature.split(',')
    feature_ids = [attribute_ids[feature] for feature in features]
    include_attributes = args.include.split(',')
    if args.exclude:
        exclude_attributes = args.exclude.split(',')
    else:
        exclude_attributes = []
    for attr in features:
        if not attr in attribute_names:
            print('Error: {} in feature is invalid attribute'.format(attr))
            exit()
    for attr in include_attributes:
        if not attr in attribute_names:
            print('Error: {} is invalid attribute'.format(attr))
            exit()
        attribute_flag[attribute_ids[attr]] = 1
    for attr in exclude_attributes:
        if not attr in attribute_names:
            print('Error: {} is invalid attribute'.format(attr))
            exit()
        attribute_flag[attribute_ids[attr]] = -1
    cols = six.moves.range(1, attribute_size + 1)
    attributes = np.loadtxt(args.attr_file, skiprows=2, usecols=cols, dtype=np.int32)
    source_paths = []
    target_paths = []
    for i, attr in enumerate(attributes):
        if not np.all(attr * attribute_flag >= 0):
            continue
        feature_flags = [attr[j] >= 0 for j in feature_ids]
        file_path = os.path.join(args.image_dir, '{0:06d}.jpg'.format(i + 1))
        if all(feature_flags):
            target_paths.append(file_path)
        elif not any(feature_flags):
            source_paths.append(file_path)
    if not args.dry_run:
        with open(args.source_list, 'w') as f:
            f.write('\n'.join(source_paths))
            f.write('\n')
        with open(args.target_list, 'w') as f:
            f.write('\n'.join(target_paths))
            f.write('\n')
    print('{} images with feature'.format(len(target_paths)))
    print('{} images without feature'.format(len(source_paths)))

if __name__ == '__main__':
    main()

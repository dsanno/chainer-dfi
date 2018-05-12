import argparse
import numpy as np
import os

from train import train

attribute_names = 'Male,Asian,White,Black,Baby,Child,Youth,Middle_Aged,Senior,Black_Hair,Blond_Hair,Brown_Hair,Bald,No_Eyewear,Eyeglasses,Sunglasses,Mustache,Smiling,Frowning,Chubby,Blurry,Harsh_Lighting,Flash,Soft_Lighting,Outdoor,Curly_Hair,Wavy_Hair,Straight_Hair,Receding_Hairline,Bangs,Sideburns,Fully_Visible_Forehead,Partially_Visible_Forehead,Obstructed_Forehead,Bushy_Eyebrows,Arched_Eyebrows,Narrow_Eyes,Eyes_Open,Big_Nose,Pointy_Nose,Big_Lips,Mouth_Closed,Mouth_Slightly_Open,Mouth_Wide_Open,Teeth_Not_Visible,No_Beard,Goatee,Round_Jaw,Double_Chin,Wearing_Hat,Oval_Face,Square_Face,Round_Face,Color_Photo,Posed_Photo,Attractive_Man,Attractive_Woman,Indian,Gray_Hair,Bags_Under_Eyes,Heavy_Makeup,Rosy_Cheeks,Shiny_Skin,Pale_Skin,5_o\'_Clock_Shadow,Strong_Nose-Mouth_Lines,Wearing_Lipstick,Flushed_Face,High_Cheekbones,Brown_Eyes,Wearing_Earrings,Wearing_Necktie,Wearing_Necklace'.lower().split(',')
attribute_ids = {k:i for i, k in enumerate(attribute_names)}

def parse_arg():
    parser = argparse.ArgumentParser('Deep Feature Interpolation using LFW dataset')
    parser.add_argument('image_dir', type=str, help='Image directory path')
    parser.add_argument('attr_file', type=str, help='Attribute file path')
    parser.add_argument('input_name', type=str, help='Input person name')
    parser.add_argument('input_index', type=int, help='Input image index')
    parser.add_argument('feature', type=str, help='Feature name. Words must be concatenated with "_", and leading with "~" indicates set feature off (e.g. ~no_beard). You can use multiple features with comma separated feature names (e.g. ~mouth_closed,smiling)')
    parser.add_argument('output_image', type=str, help='Output image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device index (negative value indicate CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg19.model', help='Model file path')
    parser.add_argument('--batch_size', '-b', type=int, default=10, help='Mini batch size')
    parser.add_argument('--lr', '-l', type=float, default=1, help='Learning rate')
    parser.add_argument('--iter', '-i', type=int, default=1000)
    parser.add_argument('--near-image', type=int, default=100, help='Maximum number of source/target images for nearest neighbor')
    parser.add_argument('--tv-weight', type=float, default=100.0, help='Total variation loss weight')
    parser.add_argument('--single-weight-mode', type=float, help='When specified runs DFI only for supplied weight')
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
    return [make_image_path(image_dir, names[i], numbers[i]) for i in indices]

def nearest_attributes(attribute_dataset, attribute, feature_flags, nearest_num, reverse=False):
    names, numbers, attributes = attribute_dataset
    indices = np.arange(len(names))
    selected = True
    for feature, on in feature_flags:
        attribute_id = attribute_ids[feature]
        if reverse:
            on = not on
        if on:
            feature_selected = attributes[:,attribute_id] >= 0.5
        else:
            feature_selected = attributes[:,attribute_id] < -0.5
        selected = np.logical_and(selected, feature_selected)
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
    features = args.feature.lower().split(',')
    feature_flags = []
    for feature in features:
        if feature.startswith('~'):
            feature = feature[1:]
            feature_flags.append((feature, False))
        else:
            feature_flags.append((feature, True))
        if not feature in attribute_names:
            print('Error: {} is invalid attribute'.format(feature))
            exit()
    source_indices = nearest_attributes(attribute_dataset, attribute, feature_flags, args.near_image, reverse=True)
    target_indices = nearest_attributes(attribute_dataset, attribute, feature_flags, args.near_image)
    source_paths = image_paths(args.image_dir, attribute_dataset, source_indices)
    target_paths = image_paths(args.image_dir, attribute_dataset, target_indices)
    image_path = make_image_path(args.image_dir, args.input_name, args.input_index)
    clip_rect = (40, 20, 210, 190)
    train(args, image_path, source_paths, target_paths, clip_rect, clip_rect)

if __name__ == '__main__':
    main()

# chainer-dfi

Implementation of "Deep Feature Interpolation for Image Content Changes"(https://arxiv.org/abs/1611.05507) using Chainer.

## Requirements

* Python 2.7
* [Chainer 1.16.0](http://chainer.org/)
* [Pillow 3.1.0](https://pillow.readthedocs.io/)

## Usage

### Download Caffe VGG-19 layer model

Download VGG_ILSVRC_19_layers.caffemodel from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77.

### Convert Caffe model to Chainer model.

```
$ python src/create_chainer_model.py
```

### Download dataset

* Download img_align_celeba.zip and list_attr_celeba.txt from [CelebA Dataset web site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
* Extract zip file.

### Extract images that has specific feature

Example:

```
$ python src/extract_image.py img_align_celeba list_attr_celeba.txt image/smile image/normal smiling young,black_hair,straight_hair -e eyeglasses,male,wearing_hat,pale_skin,narrow_eyes,bushy_eyebrows,chubby,double_chin,bald,bangs,receding_hairline,sideburns,wavy_hair
```

### Interpolate feature

Example:

```
$ python src/train.py sample/sample.png sample/out.png image/normal image/smile -g 0 -i 2000
```

## License

MIT License

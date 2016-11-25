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

### Make image list for source and target images

Example:

```
$ python src/extract_image.py img_align_celeba list_attr_celeba.txt image_normal.txt image_smile.txt smiling young,bags_under_eyes -e eyeglasses,male,pale_skin,narrow_eyes,bushy_eyebrows,chubby,double_chin,bald,bangs,receding_hairline,sideburns,wavy_hair,blond_hair,gray_hair,mouth_slightly_open
```

### Interpolate feature

Example:

```
$ python src/train.py sample/sample.png image/out/out.png image_normal.txt image_smile.txt -g 0 -c 19,39,159,179
```

## Output example

|Original|Weight: 0.4|Weight: 0.8|
|:---:|:---:|:---:|
|![Original image](/sample/sample.png "Original image")|![Image with interpolation weight=0.4](/sample/sample_w04.png "Weight: 0.4")|![Image with interpolation weight=0.8](/sample/sample_w04.png "Weight: 0.8")|

|Weight: 1.2|Weight: 1.6|Weight: 2.0|
|:---:|:---:|:---:|
|![Image with interpolation weight=1.2](/sample/sample_w12.png "Weight: 1.2")|![Image with interpolation weight=1.6](/sample/sample_w16.png "Weight: 1.6")|![Image with interpolation weight=2.0](/sample/sample_w20.png "Weight: 2.0")|

## License

MIT License

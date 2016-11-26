# chainer-dfi

Implementation of "Deep Feature Interpolation for Image Content Changes"(https://arxiv.org/abs/1611.05507) using Chainer.

# Requirements

* Python 2.7
* [Chainer 1.16.0](http://chainer.org/)
* [Pillow 3.1.0](https://pillow.readthedocs.io/)

# Usage

## Download Caffe model and convert

### Download Caffe VGG-19 layer model

Download VGG_ILSVRC_19_layers.caffemodel from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77.

### Convert Caffe model to Chainer model.

```
$ python src/create_chainer_model.py
```

## To use Labeled Faces in the Wild (LFW)

### Download dataset

* Download "All images aligned with deep funneling"(lfw-deepfunneled.tgz) from [LFW web site](http://vis-www.cs.umass.edu/lfw/)
* Download "LFW attributes file"(lfw_attributes.txt) from the same site.
* Extract tgz file.

### Interpolate feature

Example:

```
python src/train_lfw.py lfw-deepfunneled lfw_attributes.txt "Silvio Berlusconi" 23 smiling image/lfw_out.jpg -g 0
```

### Output example

person name: "Silvio Berlusconi"  
image number: 23

#### Feature: smiling

|Original|Weight: 0.1|Weight: 0.2|Weight: 0.3|Weight: 0.4|Weight: 0.5|
|:---:|:---:|:---:|:---:|:---:|:---:|
|![Original image](/sample/sample_lwf_original.jpg "Original image")|![Image with interpolation weight=0.1](/sample/sample_lwf_smiling_w01.jpg "Weight: 0.1")|![Image with interpolation weight=0.2](/sample/sample_lwf_smiling_w02.jpg "Weight: 0.2")|![Image with interpolation weight=0.3](/sample/sample_lwf_smiling_w03.jpg "Weight: 0.3")|![Image with interpolation weight=0.4](/sample/sample_lwf_smiling_w04.jpg "Weight: 0.4")|![Image with interpolation weight=0.5](/sample/sample_lwf_smiling_w05.jpg "Weight: 0.5")|

#### Feature: aged

|Original|Weight: 0.1|Weight: 0.2|Weight: 0.3|Weight: 0.4|Weight: 0.5|
|:---:|:---:|:---:|:---:|:---:|:---:|
|![Original image](/sample/sample_lwf_original.jpg "Original image")|![Image with interpolation weight=0.1](/sample/sample_lwf_aged_w01.jpg "Weight: 0.1")|![Image with interpolation weight=0.2](/sample/sample_lwf_aged_w02.jpg "Weight: 0.2")|![Image with interpolation weight=0.3](/sample/sample_lwf_aged_w03.jpg "Weight: 0.3")|![Image with interpolation weight=0.4](/sample/sample_lwf_aged_w04.jpg "Weight: 0.4")|![Image with interpolation weight=0.5](/sample/sample_lwf_aged_w05.jpg "Weight: 0.5")|

## To use Large-scale CelebFaces Attributes (CelebA) Dataset

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

### Output example

#### Feature: smiling

|Original|Weight: 0.1|Weight: 0.2|
|:---:|:---:|:---:|
|![Original image](/sample/sample.png "Original image")|![Image with interpolation weight=0.1](/sample/sample_w01.png "Weight: 0.1")|![Image with interpolation weight=0.2](/sample/sample_w02.png "Weight: 0.2")|

|Weight: 0.3|Weight: 0.4|Weight: 0.5|
|:---:|:---:|:---:|
|![Image with interpolation weight=0.3](/sample/sample_w03.png "Weight: 0.3")|![Image with interpolation weight=0.4](/sample/sample_w04.png "Weight: 0.4")|![Image with interpolation weight=0.5](/sample/sample_w05.png "Weight: 0.5")|

# License

MIT License

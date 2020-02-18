# toy-cyclegan
Simplified PyTorch implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) based on [the official implementaton](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

<br>

## Usage

#### Clone this repository
```bash
$ git clone http://github.com/jlim262/toy-cyclegan.git
$ cd toy-cyclegan
```

#### Prepare your dataset
In order to translate Bart Simpson to Lisa Simpson, 
Download Bart Simpson images into ./dataset/bart2lisa/trainA and Lisa Simpson into ./dataset/bart2lisa/trainB

#### Train a model
```bash
$ python train.py
```

#### View results on tensorboard
```bash
$ tensorboard --logdir=runs
```

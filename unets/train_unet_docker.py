import os
import sys
import time
import numpy as np
import json

np.random.seed(9999)

import image_handler as ih

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *

from cnn_tools import *
from data_tools import *

K.set_image_dim_ordering('th')


defaults = {
    "use_boss": False,
    "train_pct": 0.50,
    "n_epochs": 10,
    "mb_size": 4,
    "n_mb_per_epoch": 3,
    "save_freq": 50,
    "do_warp": False,
    "weights_file": None
}


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args(json_file=None):
    args = defaults

    if json_file:
        with open(json_file, 'r') as f:
            json_args = json.load(f)
        args.update(json_args)

    return Namespace(**args)


def get_boss_data(args):

    config = {"protocol": "https",
              "host": "api.theBoss.io",
              "token": args.token}
    rmt = BossRemote(config)

    chan = ChannelResource(args.img_channel,
                           args.collection,
                           args.experiment,
                           'image',
                           datatype='uint8')

    # Get the image data from the BOSS
    x_train = rmt.get_cutout(chan, args.resolution,
                             args.x_rng,
                             args.y_rng,
                             args.z_rng)

    lchan = ChannelResource(args.lbl_channel,
                            args.collection,
                            args.experiment,
                            'annotation',
                            datatype='uint64')

    y_train = rmt.get_cutout(lchan, args.resolution,
                             args.x_rng,
                             args.y_rng,
                             args.z_rng)

    # Data must be [slices, chan, row, col] (i.e., [Z, chan, Y, X])
    x_train = x_train[:, np.newaxis, :, :].astype(np.float32)
    y_train = y_train[:, np.newaxis, :, :].astype(np.float32)

    # Pixel values must be in [0,1]
    x_train /= 255.
    y_train = (y_train > 0).astype('float32')

    return x_train, y_train


def get_file_data(args):

    file_type = args.img_file.split('.')[-1]
    if file_type == 'gz' or file_type == 'nii':
        x_train = ih.load_nii(args.img_file, data_type='uint8')
        y_train = ih.load_nii(args.lbl_file, data_type='uint8')

    elif file_type == 'npz':
        x_train = np.load(args.img_file)
        y_train = np.load(args.lbl_file)

    # Data must be [slices, chan, row, col] (i.e., [Z, chan, Y, X])
    x_train = x_train[:, np.newaxis, :, :].astype(np.float32)
    y_train = y_train[:, np.newaxis, :, :].astype(np.float32)

    # Pixel values must be in [0,1]
    x_train /= 255.
    y_train = (y_train > 0).astype('float32')

    return x_train, y_train


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        exit('JSON config file was not provided.')

    args = parse_args(sys.argv[1])

    if args.use_boss:
        x_train, y_train = get_boss_data(args)
    else:
        x_train, y_train = get_file_data(args)

    tile_size = tuple(args.tile_size)
    train_pct = args.train_pct
    # -------------------------------------------------------------------------

    # Data must be [slices, chan, row, col] (i.e., [Z, chan, Y, X])
    # split into train and valid
    train_slices = range(int(train_pct * x_train.shape[0]))
    x_train = x_train[train_slices, ...]
    y_train = y_train[train_slices, ...]

    valid_slices = range(int(train_pct * x_train.shape[0]), x_train.shape[0])
    x_valid = x_train[valid_slices, ...]
    y_valid = y_train[valid_slices, ...]

    print('[info]: training data has shape:     %s' % str(x_train.shape))
    print('[info]: training labels has shape:   %s' % str(y_train.shape))
    print('[info]: validation data has shape:   %s' % str(x_valid.shape))
    print('[info]: validation labels has shape: %s' % str(y_valid.shape))
    print('[info]: tile size:                   %s' % str(tile_size))

    # train model
    tic = time.time()
    model = create_unet((1, tile_size[0], tile_size[1]))
    if args.do_synapse:
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=pixelwise_crossentropy_loss_w,
                      metrics=[f1_score])
    else:
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=pixelwise_crossentropy_loss,
                      metrics=[f1_score])

    if args.weights_file:
        model.load_weights(args.weights_file)

    train_model(x_train, y_train, x_valid, y_valid, model,
                args.output_dir, do_augment=args.do_warp,
                n_epochs=args.n_epochs, mb_size=args.mb_size,
                n_mb_per_epoch=args.n_mb_per_epoch,
                save_freq=args.save_freq)

    print('[info]: total time to train model: %0.2f min' %
          ((time.time() - tic) / 60.))

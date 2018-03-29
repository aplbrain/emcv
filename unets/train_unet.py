"""
   Trains a dense (per-pixel) classifier on EM data.
"""

from __future__ import print_function

__author__ = 'mjp, Nov 2016'
__license__ = 'Apache 2.0'

import os
import sys
import time
import json
import numpy as np

np.random.seed(9999)

from keras import backend as K

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *

from cnn_tools import *
from data_tools import *

K.set_image_dim_ordering('th')

if __name__ == '__main__':

    with open('/jobs/train_job_params.json') as f:
        params = json.load(f)

    # -------------------------------------------------------------------------
    rmt = BossRemote('/jobs/boss_config.cfg')

    img_chan = ChannelResource(params['img_channel'],
                               params['collection'],
                               params['experiment'],
                               type='image',
                               datatype='uint8')

    lbl_chan = ChannelResource(params['lbl_channel'],
                               params['collection'],
                               params['experiment'],
                               type='annotation',
                               datatype='uint64')

    # Get the image data from the BOSS
    x_train = rmt.get_cutout(img_chan, params['resolution'],
                             params['x_rng'],
                             params['y_rng'],
                             params['z_rng'])
    y_train = rmt.get_cutout(lbl_chan, params['resolution'],
                             params['x_rng'],
                             params['y_rng'],
                             params['z_rng'])

    # Data must be [slices, chan, row, col] (i.e., [Z, chan, Y, X])
    x_train = x_train[:, np.newaxis, :, :].astype(np.float32)
    y_train = y_train[:, np.newaxis, :, :].astype(np.float32)
    # Pixel values must be in [0,1]
    x_train /= 255.
    y_train = (y_train > 0).astype('float32')

    tile_size = tuple(params['tile_size'])
    train_pct = params['train_pct']
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
    if params['do_synapse']:
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=pixelwise_crossentropy_loss_w,
                      metrics=[f1_score])
    else:
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=pixelwise_crossentropy_loss,
                      metrics=[f1_score])

    # if weights_file:
    #     model.load_weights(weights_file)

    train_model(x_train, y_train, x_valid, y_valid, model,
                params['output_dir'], do_augment=params['do_augment'],
                n_epochs=params['n_epochs'], mb_size=params['mb_size'],
                n_mb_per_epoch=params['n_mb_per_epoch'],
                save_freq=params['save_freq'])

    print('[info]: total time to train model: %0.2f min' %
          ((time.time() - tic)/60.))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

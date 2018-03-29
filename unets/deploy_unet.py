import os
import sys
import time
import numpy as np
import json
import argparse

np.random.seed(9999)

from keras import backend as K
from keras.models import load_model

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *

from cnn_tools import *
from data_tools import *

K.set_image_dim_ordering('th')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', dest='collection', type=str,
                        help='BOSS Collection')
    parser.add_argument('--experiment', dest='experiment', type=str,
                        help='BOSS Experiment')
    parser.add_argument('--in_channel', dest='in_channel', type=str,
                        help='BOSS Image Channel')
    parser.add_argument('--out_channel', dest='out_channel', type=str,
                        help='BOSS output channel')
    parser.add_argument('--x_rng', dest='x_rng', type=str,
                        help='Volume X range (comma-separated min,max)')
    parser.add_argument('--y_rng', dest='y_rng', type=str,
                        help='Volume Y range (comma-separated min,max)')
    parser.add_argument('--z_rng', dest='z_rng', type=str,
                        help='Volume Z range (comma-separated min,max)')
    parser.add_argument('--res', dest='resolution', type=int,
                        help='BOSS image resolution level')
    parser.add_argument('--z_step', dest='z_step', type=int, default=1,
                        help='Amount to step in Z when performing inference')
    parser.add_argument('--do_synapse', dest='do_synapse', type=bool,
                        default=False,
                        help='Flag to toggle between mem/syn prediction')
    parser.add_argument('--token', dest='token', type=str,
                        help='BOSS token')

    args = parser.parse_args()

    # Ranges are initially mapped to comma-separated values
    args.x_rng = list(map(int, args.x_rng.split(',')))
    args.y_rng = list(map(int, args.y_rng.split(',')))
    args.z_rng = list(map(int, args.z_rng.split(',')))

    return args

if __name__ == '__main__':
    # ----------------------------------------------------------------------
    args = parse_args()
    config = {"protocol": "https",
              "host": "api.theBoss.io",
              "token": args.token}
    rmt = BossRemote(config)

    chan = ChannelResource(args.in_channel,
                           args.collection,
                           args.experiment,
                           'image',
                           datatype='uint8')

    # Get the image data from the BOSS
    x_test = rmt.get_cutout(chan, args.resolution,
                            args.x_rng,
                            args.y_rng,
                            args.z_rng)

    # Data must be [slices, chan, row, col] (i.e., [Z, chan, Y, X])
    x_test = x_test[:, np.newaxis, :, :].astype(np.float32)
    # Pixel values must be in [0,1]
    if x_test.max() > 1.0:
        x_test /= 255.

    tile_size = (512, 512)
    z_step = args.z_step
    # ----------------------------------------------------------------------

    # load model
    model = create_unet((1, tile_size[0], tile_size[1]))
    if args.do_synapse:
        model.load_weights('/src/weights/synapse_weights.hdf5')
    else:
        model.load_weights('/src/weights/membrane_weights.hdf5')

    tic0 = time.time()
    tic = time.time()
    y_hat = np.zeros(x_test.shape)
    for i in range(0, x_test.shape[0], z_step):
        y_hat[i:i+z_step, ...] = deploy_model(x_test[i:i+z_step, ...], model)

    print('[info]: total time to deploy: %0.2f sec' % (time.time() - tic))
    y_hat = (255. * np.squeeze(y_hat)).astype(np.uint8)

    print('Total time to process entire volume: {}'.format(time.time() - tic0))

    # ----------------------------------------------------------------------
    # print('Saving npz file...')
    # np.savez(args.output_file, y_hat=y_hat,
    #          x_test=(255 * x_test).astype(np.uint8))
    # print('Done.')
    out_chan = ChannelResource(args.out_channel,
                               args.collection,
                               args.experiment,
                               'annotation',
                               datatype='uint8')
    rmt.create_cutout(out_chan, args.resolution,
                      args.x_rng,
                      args.y_rng,
                      args.z_rng, y_hat)
    # ----------------------------------------------------------------------

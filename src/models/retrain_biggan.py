import os
import sys
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import pytorch_biggan
# this is a lil trick to fool PyCharm
try:
    import models.pytorch_biggan.inception_utils as inception_utils
    import models.pytorch_biggan.utils as utils
    import models.pytorch_biggan.losses as losses
    import models.pytorch_biggan.train_fns as train_fns
    from models.pytorch_biggan.datasets import ImageFolder
    from models.pytorch_biggan.sync_batchnorm import patch_replication_callback
except:
    sys.path.append('models/pytorch_biggan/')
    import inception_utils
    import utils
    import losses
    import train_fns
    from datasets import ImageFolder
    from sync_batchnorm import patch_replication_callback


import src.models.trimmed_biggan as trimmed_biggan

pretrained_folder = 'models/pytorch_biggan/pretrained/100k/'
CONFIG = {
    'dataset': 'met',
    # 'dataset': 'I128_hdf5',
    'resume': True,
    'resume_weights': pretrained_folder,
    'num_workers': 1,
    'shuffle': True,
    'load_in_mem': True,
    'use_multiepoch_sampler': True,
    'G_ch': 96,
    'D_ch': 96,
    'G_shared': True,
    'shared_dim': 128,
    'dim_z': 120,
    'hier': True,
    'G_nl': 'inplace_relu',
    'D_nl': 'inplace_relu',
    'G_lr': 0.0001,
    'D_lr': 0.0004,
    'batch_size': 16,
    'num_D_steps': 1,
    'num_epochs': 1,
    'G_eval_mode': True,
    'save_every': 10,
    'num_best_copies': 5,
    'test_every': 10,
    'data_root': '.',
    'weights_root': 'models/retrained/',
    'logs_root': 'output/',
    'samples_root': 'output/',
    'experiment_name': 'retrained/',
    'ema': True,
    'use_ema': True,
    'ema_start': 20000,
    'adam_eps': 1e-06,
    'SN_eps': 1e-06
}


def test_pretrain_sample(G, config):
    print('creating test image')
    gen_batch = 1
    # z_, y_ = utils.prepare_z_y(gen_batch, config['dim_z'], config['n_classes'])
    # utils.seed_rng(0)
    z_, y_ = utils.prepare_z_y(gen_batch, config['dim_z'], 1000)
    z_.sample_()
    y_.sample_()
    # print(z_.shape, y_.shape)
    # input()
    with torch.no_grad():
        G_z = G(z_, G.shared(y_))
    image = np.uint8(255 * (G_z.cpu().numpy() + 1) / 2.)
    image = np.transpose(image[0], (1, 2, 0))
    plt.imshow(image)
    plt.show()


def sample_gen(G, config):
    pass


# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']] = 1000
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)

    # Next, build the model
    G = trimmed_biggan.Generator(**config).to(device)
    # D = model.Discriminator(**config).to(device)

    # remove dimensions that will differ
    for blocklist in G.blocks:
        for block in blocklist:
            print('block', block)
            # input()
    # input('end blocks')
    del G.shared

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init': True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
        del G_ema.shared
    else:
        G_ema, ema = None, None

    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()

    # print(G)
    print('Number of params in G: {}'.format(
        *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, ]]))
    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                  'best_IS': 0, 'best_FID': 999999, 'config': config}

    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        utils.load_weights(G, None, state_dict,
                           config['resume_weights'], '',
                           config['load_weights'] if config['load_weights'] else None,
                           G_ema if config['ema'] else None, strict=False)
        # make sure it starts counting from 0
        state_dict['itr'] = 0
        state_dict['epoch'] = 0


    # can only do this if we remove 'del G.shared' above
    # test_pretrain_sample(G, config)
    config['input_shape'] = (128, 128)

    x = np.random.random(config['input_shape'])
    y = G(x)

    def img_gen():
        for f in os.listdir('data/processed/'):
            print('data file', f)

    img_gen()
    def train_step(x, y):
        G.optim.zero_grad()


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    # print(config)
    # run(config)
    # CONFIG was constructed with the values that differ from the defaults
    # thus, no command line args need to be passed
    config.update(CONFIG)
    run(config)


if __name__ == '__main__':
    # vase_img_dataset = ImageFolder(data_folder)
    main()

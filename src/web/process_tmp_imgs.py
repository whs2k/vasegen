import os
import sys
import time

orig = sys.argv
sys.argv = [sys.argv[0]]
sys.argv+= '--dataroot ' \
           '../../data/processed/pix2pix_vase_examples_512 ' \
           '--name pix2pix_vase_fragments_512 ' \
           '--model pix2pix --netG unet_512 --direction BtoA --dataset_mode aligned --norm batch ' \
           '--eval --preprocess none --gpu_ids -1'.split()

os.chdir('models/pix2pix/')
sys.path.append('.')
from models import create_model
from options.test_options import TestOptions
from data.base_dataset import get_params, get_transform

import numpy as np
import torch
import torchvision
from PIL import Image

import glob
from tempfile import gettempdir
from multiprocessing import Process
TMPDIR = gettempdir() + '/vasegen/'
MAX_JOBS = 1
jobs = {}
class DummyJob:
    exitcode = 0

# opt = None
opt = TestOptions().parse()  # get test options

# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

to_pil = torchvision.transforms.ToPILImage()
to_tensor = torchvision.transforms.ToTensor()

def load_img(file):
    return Image.open(file.stream).convert('RGB')

def create_img_dict_single(A, B=None):
    if type(A) is torch.Tensor:
        A = to_pil(A)

    # apply the same transform to both A and B
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=False)
    A = A_transform(A).unsqueeze(0)
    if B is not None:
        B_transform = get_transform(opt, transform_params, grayscale=False)
        B = B_transform(B).unsqueeze(0)
    else:
        B = torch.randn(1, 3, A.shape[2], A.shape[3])

    return {'A': B, 'B': A, 'A_paths': '', 'B_paths': ''}

def create_img_dict(AB):
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))

    return create_img_dict_single(A, B)


def process_img(model, data):
    model.set_input(data)                  # unpack data from data loader
    model.test()                           # run inference
    visuals = model.get_current_visuals()  # get image results
    image_numpy = visuals['fake_B'][0].numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def predict(model, fname):
    outname = fname.replace('A_', 'B_')
    outtouch = outname.replace('.png', '')
    time.sleep(.1)
    img = Image.open(fname).convert('RGB')
    # img.show()
    # print(img.info)
    # print(img.mode)
    # input()

    img_data = create_img_dict_single(img)
    res = process_img(model, img_data)
    res = Image.fromarray(res)
    res.save(outname)
    os.system(f'touch {outtouch}')
    return 0


def process_imgs(model):
    while True:
        time.sleep(.1)
        print(time.time(), 'jobs dict', jobs)
        fnames = glob.glob(TMPDIR + f'A_*.png')
        for fname in fnames:
            if MAX_JOBS == 1:
                predict(model, fname)
                jobs[fname] = DummyJob
            else:
                if fname not in jobs and len(jobs) < MAX_JOBS:
                    jobs[fname] = Process(target=predict, args=(model, fname,))
                    jobs[fname].start()

        for fname in list(jobs.keys()):
            if jobs[fname].exitcode is not None:
                os.remove(fname)
                del jobs[fname]



if __name__ == '__main__':
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    os.chdir('../..')

    # for fname in glob.glob(TMPDIR + f'A_*.png'):
    #     predict(model, fname)
    process_imgs(model)

"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
# import io, base64
import os
import sys

orig = sys.argv
sys.argv = [sys.argv[0]]
sys.argv+= '--dataroot ' \
           '../../data/processed/pix2pix_vase_examples_512 ' \
           '--name pix2pix_vase_fragments_512 ' \
           '--model pix2pix --netG unet_512 --direction BtoA --dataset_mode aligned --norm batch ' \
           '--eval --preprocess none'.split()

# print(len(orig) == len(sys.argv))
# for n in range(len(orig)):
#     print(orig[n]==sys.argv[n])
# input()

os.chdir('models/pix2pix/')
sys.path.append('.')
from models import create_model
from options.test_options import TestOptions
from data.base_dataset import get_params, get_transform

# from data import create_dataset
# from util.visualizer import save_images
# from util import html

import numpy as np
import torch
import torchvision
from PIL import Image

from flask import Flask, jsonify, request

app = Flask(__name__)
opt = None


to_pil = torchvision.transforms.ToPILImage()
to_tensor = torchvision.transforms.ToTensor()

def load_img(file):
    return Image.open(file.stream).convert('RGB')

def create_img_dict_single(A, B=None):
    if B is None:
        B = to_pil(torch.zeros(3, A.size[0], A.size[1]))

    # apply the same transform to both A and B
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=False)
    B_transform = get_transform(opt, transform_params, grayscale=False)

    A = A_transform(A).unsqueeze(0)
    B = B_transform(B).unsqueeze(0)

    return {'A': A, 'B': B, 'A_paths': '', 'B_paths': ''}

def create_img_dict(AB):
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))

    return create_img_dict_single(A, B)

def img_to_bin(img):
    # input(to_pil(img).tobytes()[:20])
    return to_pil(img).tobytes().decode('iso-8859-1')
    # return str(to_pil(img).tobytes())

@app.route('/')
def home():
    return jsonify({'success': True})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['img']
    try:
        img = load_img(file)
    except:
        img = None

    if img is not None:
        # img_tensor = to_tensor(img)
        # input_tensor = transform_image(file)

        img_data = create_img_dict(img)
        res = process_img(img_data)
        # Image.fromarray(res).show()

        return jsonify({'result': img_to_bin(res), 'mode': 'RGB', 'size': (512, 512)})
    else:
        return jsonify({'result': '-1'})


def process_img(data):
    model.set_input(data)                  # unpack data from data loader
    model.test()                           # run inference
    visuals = model.get_current_visuals()  # get image results
    image_numpy = visuals['fake_B'][0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def run_app():
    app.run(debug=True)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    os.chdir('../..')
    run_app()

import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch_dct as dct
import matplotlib.pyplot as plt
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)


from src.models.utils import FragmentDataset, PreGAN, BothGAN, \
    loss_fn_scaled_mse, loss_fn_scaled_mae, sobel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def generate(model):
    # Prepare a input
    truncation = 0.4
    batch_size = 10
    class_vector = one_hot_from_names(['vase'], batch_size=batch_size)
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')
    model.to('cuda')

    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    # If you have a sixtel compatible terminal you can display the images in the terminal
    # (see https://github.com/saitoha/libsixel for details)
    # display_in_terminal(output)

    # Save results as png images
    save_as_images(output, file_name='output/output')

def retrain(vaseGen, dataset, N, batch_size=1):
    vaseGen.pregan.train()
    # vaseGen.biggan.eval()
    vaseGen.biggan.train()
    # loss_fn_mse = nn.MSELoss()
    # loss_fn_dct = lambda x, y: loss_fn_mse(dct.dct(x), dct.dct(y))
    # loss_fn_both = lambda x, y: loss_fn_mse(x, y) + loss_fn_dct(x, y)*1e-7
    # loss_fn_both = lambda x, y: loss_fn_mse(x, y) * loss_fn_dct(x, y)

    # TODO: use the resulting weights in latent space and interpolate vases
    try:
        # for n, (x, y) in enumerate(dataset.take(N, batch_size)):
        x, y = next(dataset.take(1, batch_size))
        for n in range(1000):
            if n == N // 2:
                pass
                # vaseGen.biggan.train()
            vaseGen.optim.zero_grad()
            x = x.to('cuda')
            y = y.to('cuda')
            y_pred = vaseGen(x)
            # y_edge = sobel(y).cpu().numpy()[0, 0, :, :]
            # plt.imshow(y_edge)
            # plt.show()
            loss1 = loss_fn_scaled_mse(y_pred, y)
            loss2 = None
            loss = loss1

            # loss2 = loss_fn_scaled_mse(sobel(y_pred), sobel(y))
            # loss = loss1 + loss2
            if torch.isnan(loss):
                break
            loss.backward()
            vaseGen.optim.step()
            print('step', n, 'loss1', loss1, 'loss2', loss2)
    except:
        print('training exited early')

    with torch.no_grad():
        test_vase = vaseGen(x)
    test_vase = test_vase.to('cpu').numpy()
    vase = y
    print('max', np.max(test_vase), 'min', np.min(test_vase))
    vase = vase.cpu().numpy()
    plt.subplot(121)
    plt.imshow(test_vase[0].transpose((1,2,0)))
    plt.subplot(122)
    plt.imshow(vase[0].transpose((1,2,0)))
    plt.suptitle('Last Training Sample')
    plt.show()

def vase_generate(vaseGen, data_gen):
    vaseGen.eval()
    for frag, vase in data_gen.take(1, 1):
        frag = frag.to('cuda')
        with torch.no_grad():
            # test_vase = pregan(frag)
            test_vase = vaseGen(frag)
        test_vase = test_vase.to('cpu').numpy()
        print('max', np.max(test_vase), 'min', np.min(test_vase))
        vase = vase.numpy()
        plt.subplot(121)
        plt.imshow(test_vase[0].transpose((1,2,0)))
        plt.subplot(122)
        plt.imshow(vase[0].transpose((1,2,0)))
        plt.suptitle('Generated and Target Vase')
        plt.show()


def main():
    pregan = PreGAN()
    pregan.init_weights()
    # print(pregan)
    # pregan.to('cuda')  # done by vaseGen

    biggan = BigGAN.from_pretrained('biggan-deep-512')
    # generate(biggan)
    # biggan.to('cuda')  # done by vaseGen

    vaseGen = BothGAN(pregan, biggan, lr=1e-5)
    vaseGen.to('cuda')

    data_gen = FragmentDataset()

    # vase_generate(vaseGen, data_gen)

    batch_size = 1
    n_samples = 100
    retrain(vaseGen, data_gen, n_samples, batch_size)
    while True:
        vase_generate(vaseGen, data_gen)


if __name__ == '__main__':
    main()

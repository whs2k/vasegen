import glob
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)


from src.models.utils import FragmentDataset, PreGAN, BothGAN

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
    for n, (x, y) in enumerate(dataset.take(N, batch_size)):
        vaseGen.pregan.optim.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')
        loss = nn.MSELoss()(vaseGen(x), y)
        loss.backward()
        vaseGen.pregan.optim.step()
        print('step', n, 'loss', loss)

def vase_generate(vaseGen, data_gen):
    for frag, vase in data_gen.take(1, 1):
        frag = frag.to('cuda')
        with torch.no_grad():
            # test_vase = pregan(frag)
            test_vase = vaseGen(frag)
        test_vase = test_vase.to('cpu').numpy()
        vase = vase.numpy()
        plt.subplot(121)
        plt.imshow(test_vase[0].transpose((1,2,0)))
        plt.subplot(122)
        plt.imshow(vase[0].transpose((1,2,0)))
        plt.show()


def main():
    pregan = PreGAN()
    pregan.init_weights()
    # print(pregan)
    # pregan.to('cuda')  # done by vaseGen

    biggan = BigGAN.from_pretrained('biggan-deep-512')
    # generate(biggan)
    # biggan.to('cuda')  # done by vaseGen

    vaseGen = BothGAN(pregan, biggan)
    vaseGen.to('cuda')

    data_gen = FragmentDataset()

    # vase_generate(vaseGen, data_gen)

    batch_size = 2
    n_samples = 1000
    retrain(vaseGen, data_gen, n_samples, batch_size)
    while True:
        vase_generate(vaseGen, data_gen)


if __name__ == '__main__':
    main()

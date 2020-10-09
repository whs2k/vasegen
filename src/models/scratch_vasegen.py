import glob
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


from src.models.utils import FragmentDataset, ScratchGAN, loss_fn_scaled_mse


def retrain(scratchgan, dataset, N, batch_size=1):
    scratchgan.train()
    # for n, (x, y) in enumerate(dataset.take(N, batch_size)):
    x, y = next(dataset.take(1, batch_size))
    for n in range(100):
        scratchgan.optim.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')
        loss = nn.MSELoss()(scratchgan(x), y)
        # loss = loss_fn_scaled_mse(scratchgan(x), y)
        loss.backward()
        scratchgan.optim.step()
        print('step', n, 'loss', loss)

def vase_generate(scratchgan, data_gen):
    for frag, vase in data_gen.take(1, 1):
        frag = frag.to('cuda')
        with torch.no_grad():
            # test_vase = pregan(frag)
            test_vase = scratchgan(frag)
        test_vase = test_vase.to('cpu').numpy()
        print('max', np.max(test_vase), 'min', np.min(test_vase))
        vase = vase.numpy()
        plt.subplot(121)
        plt.imshow(test_vase[0].transpose((1,2,0)))
        plt.subplot(122)
        plt.imshow(vase[0].transpose((1,2,0)))
        plt.show()


def main():
    scratchgan = ScratchGAN()
    scratchgan.to('cuda')

    data_gen = FragmentDataset()

    # vase_generate(vaseGen, data_gen)

    batch_size = 2
    n_samples = 1000
    retrain(scratchgan, data_gen, n_samples, batch_size)
    while True:
        vase_generate(scratchgan, data_gen)


if __name__ == '__main__':
    main()

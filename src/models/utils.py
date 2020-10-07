import random
import glob

import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from torchvision import transforms, set_image_backend
# set_image_backend('accimage')

from models.pytorch_biggan.datasets  import default_loader
from pytorch_pretrained_biggan import one_hot_from_names

data_dir = 'data/processed/vase_fragment_dataset/'
full_img = lambda img_id: f'{data_dir}/full_{img_id}.jpg'
frag_img = lambda img_id, n_frag: f'{data_dir}/frag_{img_id}_{n_frag}.jpg'


def vase_vector(batch_size):
    return one_hot_from_names(['vase'], batch_size=batch_size)


def gather_ids():
    img_ids = list()
    for f in glob.glob(f'{data_dir}/full_*.jpg'):
        img_id = f.split('_')[-1].split('.')[0]
        img_ids.append(int(img_id))
    assert img_ids

    # print('num frags', n_frags)
    # print('num ids', len(img_ids))
    # print(img_ids[:10])

    n_frags = len(glob.glob(f'{data_dir}/frag_{img_ids[0]}_*.jpg'))
    return img_ids, n_frags

class FragmentDataset:
    def __init__(self):
        img_ids, n_frags = gather_ids()
        self.to_tensor = transforms.ToTensor()
        self.data_dir = data_dir
        self.img_ids = img_ids
        self.n_frags = n_frags
        self.loader = default_loader

    def take(self, N, batch_size=1):
        for _ in range(N):
            imgs, frags = [], []
            for _ in range(batch_size):
                img_id = random.choice(self.img_ids)
                # print(img_id)
                n_frag = random.randint(0, self.n_frags-1)
                img = self.loader(full_img(img_id))
                frag = self.loader(frag_img(img_id, n_frag))
                imgs += [self.to_tensor(img).unsqueeze(0)]
                frags += [self.to_tensor(frag).unsqueeze(0)]
            imgs = torch.cat(imgs, axis=0)
            frags = torch.cat(frags, axis=0)
            yield frags, imgs


class PreGAN(nn.Module):
    def __init__(self):
        super(PreGAN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(3, 16, 3)
        # self.conv2 = nn.Conv2d(16, 16, 3)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(120*120, 128)  # 6*6 from image dimension
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        #
        # mods = [
        #     self.conv1,
        #     self.conv2,
        #     self.fc1,
        #     self.fc2,
        #     self.fc3,
        # ]
        # self.layers = nn.ModuleList(mods)
        # OLD FORWARD
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # nn.Flatten(),
            nn.Linear(120 * 120, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # added custom
        self.init = 'ortho'
        self.param_count = 0

        # optim called last
        self.optim = optim.Adam(self.parameters())
        # self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
        #                         betas=(self.B1, self.B2), weight_decay=0,
        #                         eps=self.adam_eps)

    def forward(self, x):
        return self.main(x)

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for PreGAN initialized parameters: %d' % self.param_count)


class BothGAN(nn.Module):
    def __init__(self, pregan, biggan):
        super(BothGAN, self).__init__()
        self.pregan = pregan
        self.biggan = biggan
        self.vase_vec = torch.from_numpy(vase_vector(1))
        self.add_module('pregan', self.pregan)
        self.add_module('biggan', self.biggan)

    def forward(self, frag):
        noise = self.pregan(frag)
        vase_vec = torch.cat([self.vase_vec]*noise.shape[0], dim=0)
        return self.biggan(noise, vase_vec, 1.0)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.vase_vec = self.vase_vec.to(*args, **kwargs)

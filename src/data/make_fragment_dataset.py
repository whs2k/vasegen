import os
import sys
import glob
from scipy.ndimage import binary_fill_holes
import itertools
import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt


sin = lambda ang: np.sin(ang * np.pi / 180)
cos = lambda ang: np.cos(ang * np.pi / 180)
tan = lambda ang: sin(ang) / cos(ang)

def contiguous(point, shape, range=1):
    # p_x = [point[0]-1, point[0], point[0]+1]
    # p_y = [point[1]-1, point[1], point[1]+1]

    p_x = np.ones((2*range+1), dtype=np.int32)*point[0]
    p_x += np.arange(-range, range+1)

    p_y = np.ones((2*range+1), dtype=np.int32)*point[1]
    p_y += np.arange(-range, range+1)

    p_x = [p for p in p_x if 0 <= p < shape[0]]
    p_y = [p for p in p_y if 0 <= p < shape[1]]
    points = list(itertools.product(p_x, p_y))
    return points


def space_fill(img, start=None, ):
    if start is None:
        start = img.shape[0] // 2, img.shape[1] // 2

    # mask = np.zeros(img.shape, dtype=int) - 1
    # max_count = 10
    # mask[start] = max_count

    mask = np.zeros(img.shape, dtype=bool)
    thresh = np.percentile(img.flatten(), 95)

    details = img > thresh
    mask[details] = 1

    trim = 25
    mask[:trim, :] = 0
    mask[-trim:, :] = 0
    mask[:, :trim] = 0
    mask[:, -trim:] = 0

    mask_inds = np.argwhere(mask)
    m_min = np.min(mask_inds[:, 0])
    m_max = np.max(mask_inds[:, 0])
    n_min = np.min(mask_inds[:, 1])
    n_max = np.max(mask_inds[:, 1])
    print(m_min, m_max, n_min, n_max)
    return mask, m_min, m_max, n_min, n_max

    # this takes awhile, I can do simpler
    # max_range = 10
    # final_mask = np.zeros(img.shape, dtype=bool)
    # for m, n in np.ndindex(mask.shape):
    #     for i, j in contiguous((m, n), mask.shape, max_range):
    #         if mask[i, j]:
    #             final_mask[m, n] = 1
    #             break

    # final_mask = binary_fill_holes(final_mask)
    # return final_mask

def mark_image_box(img, m_min, m_max, n_min, n_max):
    new_img = np.copy(img)
    thick=5
    for m in m_min, m_max:
        # new_img[m-thick:m+thick, n_min:n_max, :] = (255, 0, 0)
        new_img[m-thick:m+thick, n_min:n_max] = 255
    for n in n_min, n_max:
        # new_img[m_min:m_max, n-thick:n+thick, :] = (255, 0, 0)
        new_img[m_min:m_max, n-thick:n+thick] = 255
    return new_img

def fragment(img, m_min, m_max, n_min, n_max):
    if m_min > m_max - frag_size or n_min > n_max - frag_size:
        return None

    # m_start = np.random.randint(m_min, m_max+1-frag_size)
    # n_start = np.random.randint(n_min, n_max+1-frag_size)

    # use normal dist, stdev range/2/2
    norm_scale = 2

    m_choices = np.arange(m_min, m_max+1-frag_size)
    n_choices = np.arange(n_min, n_max+1-frag_size)

    m_ind = np.random.normal(len(m_choices)/2, len(m_choices)/2/norm_scale)
    n_ind = np.random.normal(len(n_choices)/2, len(n_choices)/2/norm_scale)

    m_ind = round(m_ind)
    n_ind = round(n_ind)

    m_ind = 0 if m_ind < 0 else m_ind
    m_ind = len(m_choices) - 1 if m_ind >= len(m_choices) else m_ind

    n_ind = 0 if n_ind < 0 else n_ind
    n_ind = len(n_choices) - 1 if n_ind >= len(n_choices) else n_ind

    m_start = m_choices[m_ind]
    n_start = n_choices[n_ind]

    new_img = np.copy(img[m_start:m_start+frag_size, n_start:n_start+frag_size])

    n_cuts = np.random.choice([3, 4, 5], p=[.4, .4, .2])
    angles = np.linspace(0, 360, n_cuts+1)[:-1]
    # perturb each by a little
    angles += np.random.uniform(0, 30/n_cuts, angles.shape)
    # rotate all angles randomly
    angles += np.random.random()*90/n_cuts
    cut_start_m = np.random.randint(0, frag_size//2)
    cut_start_n = 0
    cuts = [[cut_start_m, cut_start_n]]
    for ang in angles:
        cut_end_m1 = 127 if np.sign(cos(ang)) > 0 else 0
        cut_end_n1 = cut_start_n + (cut_end_m1-cut_start_m) * tan(ang)
        cut_end_n2 = 127 if np.sign(sin(ang)) > 0 else 0
        cut_end_m2 = cut_start_m + (cut_end_n2-cut_start_n) / tan(ang)
        # print()
        # print('start', cut_start_m, cut_start_n)
        # print('option 1', cut_end_m1, cut_end_n1)
        # print('option 2', cut_end_m2, cut_end_n2)
        if cut_end_n1 < 0 or cut_end_n1 >= 128:
            cut_end_n = cut_end_n2
            cut_end_m = cut_end_m2
        else:
            cut_end_n = cut_end_n1
            cut_end_m = cut_end_m1

        cuts += [[cut_end_m, cut_end_n]]
        cut_start_m = cut_end_m
        cut_start_n = cut_end_n

    mask = np.ones(new_img.shape, dtype=bool)
    print('angles', angles)
    print('cuts', cuts)
    for n, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        a, b = np.array(a), np.array(b)
        print(a, b)
        ang = angles[n]
        # plt.plot((a[0], b[0]), (a[1], b[1]), marker='o')
        for t in np.linspace(0, 1, 1000):
            p = t * a + (1 - t) * b
            img_ind = np.round(p).astype(np.int)
            new_img[img_ind[0], img_ind[1], :] = (255, 0, 0)
            end_m = 128 if np.sign(sin(ang)) > 0 else 0
            end_n = 128 if np.sign(-cos(ang)) > 0 else 0
            m_slice = slice(img_ind[0], end_m) \
                if end_m > img_ind[0] else \
                slice(end_m, img_ind[0])
            n_slice = slice(img_ind[1], end_n) \
                if end_n > img_ind[1] else \
                slice(end_n, img_ind[1])
            mask[m_slice, img_ind[1]] = 0
            mask[img_ind[0], n_slice] = 0

    plt.subplot(121)
    plt.imshow(new_img)

    plt.subplot(122)
    new_img[mask] = 0
    plt.imshow(mask.astype(np.int)*255)
    plt.show()
    return new_img


dir_in = f'data/processed/full_vases/'
dir_out = f'data/processed/vase_fragment_dataset/'
n_fragments = 10
frag_size = 128
def main():
    for f_img in glob.glob(dir_in + '/*'):
        img = dip.imread(f_img)
        print(f_img, img.shape)
        if len(img.shape) == 3:
            gray = np.mean(img, axis=-1)
        else:
            # gray = img
            continue
        grad = dip.transforms.edge_detect(gray)
        mask, m_min, m_max, n_min, n_max = space_fill(grad)
        grad = mark_image_box(grad, m_min, m_max, n_min, n_max)

        frag_failed = False
        for n in range(n_fragments):
            frag = fragment(img, m_min, m_max, n_min, n_max)
            if frag is None:
                frag_failed = True
        input('done fragments')

        if frag_failed:
            continue

        plt.close('all')
        plt.subplot(221)
        plt.imshow(img)

        plt.subplot(222)
        plt.imshow(grad)

        plt.subplot(223)
        plt.imshow(255*mask)

        plt.show()
        # plt.pause(1)

if __name__ == '__main__':
    main()

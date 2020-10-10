import os
import glob
import shutil
# from scipy.ndimage import binary_fill_holes
import itertools
import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt

from tqdm import tqdm

import cv2
from cv2 import INTER_AREA
from skimage import filters
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import binary_erosion


sin = lambda ang: np.sin(ang * np.pi / 180)
cos = lambda ang: np.cos(ang * np.pi / 180)
tan = lambda ang: sin(ang) / cos(ang)
default_frag_size = (128, 128)

dir_in = f'data/processed/full_vases/'
dir_out = f'data/processed/vase_fragment_dataset/'
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

out_img = lambda img_id: f'{dir_out}/full_{img_id}.jpg'
out_frag = lambda img_id, n_frag: f'{dir_out}/frag_{img_id}_{n_frag}.jpg'

n_fragments = 10

_pix2pix_counter = 1
_pix2pix_marker_size = 5
_pix2pix_outsize = (256, 256)
_pix2pix_dir = 'data/processed/pix2pix_vase_fragments/train/'
os.makedirs(_pix2pix_dir, exist_ok=True)


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

    thresh = np.percentile(img, 95)

    # mask2 = np.zeros(img.shape, dtype=bool)
    # details = img > thresh
    # mask2[details] = 1
    mask = img > thresh

    # trim = 25
    # make this general to image shape, not just 512x512
    trim = img.shape[0] // 20
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

def fragment_slow(img, m_min, m_max, n_min, n_max, frag_size=default_frag_size):
    if m_min > m_max - frag_size[0] or n_min > n_max - frag_size[1]:
        return None, (0, 0)

    # m_start = np.random.randint(m_min, m_max+1-frag_size)
    # n_start = np.random.randint(n_min, n_max+1-frag_size)

    # use normal dist, stdev range/2/2
    norm_scale = 2

    m_choices = np.arange(m_min, m_max+1-frag_size[0])
    n_choices = np.arange(n_min, n_max+1-frag_size[1])

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

    new_img = np.copy(img[m_start:m_start+frag_size[0], n_start:n_start+frag_size[1]])

    n_cuts = np.random.choice([3, 4, 5], p=[.4, .4, .2])
    angles = np.linspace(0, 360, n_cuts+1)[:-1]
    # perturb each by a little
    angles += np.random.uniform(0, 30/n_cuts, angles.shape)
    # rotate all angles randomly
    angles += np.random.random()*90/n_cuts
    cut_start_m = np.random.randint(0, frag_size[0]//2)
    cut_start_n = 0
    cuts = [[cut_start_m, cut_start_n]]
    for ang in angles:
        cut_end_m1 = frag_size[0]-1 if np.sign(cos(ang)) > 0 else 0
        cut_end_n1 = cut_start_n + (cut_end_m1-cut_start_m) * tan(ang)
        cut_end_n2 = frag_size[1]-1 if np.sign(sin(ang)) > 0 else 0
        cut_end_m2 = cut_start_m + (cut_end_n2-cut_start_n) / tan(ang)
        # print()
        # print('start', cut_start_m, cut_start_n)
        # print('option 1', cut_end_m1, cut_end_n1)
        # print('option 2', cut_end_m2, cut_end_n2)
        if cut_end_n1 < 0 or cut_end_n1 >= frag_size[1]:
            cut_end_n = cut_end_n2
            cut_end_m = cut_end_m2
        else:
            cut_end_n = cut_end_n1
            cut_end_m = cut_end_m1

        cuts += [[cut_end_m, cut_end_n]]
        cut_start_m = cut_end_m
        cut_start_n = cut_end_n

    mask = np.ones(new_img.shape, dtype=bool)
    # print('angles', angles)
    # print('cuts', cuts)
    for n, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        a, b = np.array(a), np.array(b)
        # print(a, b)
        ang = angles[n]
        # plt.plot((a[0], b[0]), (a[1], b[1]), marker='o')
        for t in np.linspace(0, 1, max(frag_size)*2):
            p = t * a + (1 - t) * b
            img_ind = np.round(p).astype(np.int)
            img_ind[img_ind < 0] = 0

            # img_ind[img_ind > frag_size-1] = frag_size-1
            img_ind[0] = min(img_ind[0], frag_size[0]-1)
            img_ind[1] = min(img_ind[1], frag_size[1]-1)

            end_m = frag_size[0]-1 if np.sign(sin(ang)) > 0 else 0
            end_n = frag_size[0]-1 if np.sign(-cos(ang)) > 0 else 0
            m_slice = slice(img_ind[0], end_m) \
                if end_m > img_ind[0] else \
                slice(end_m, img_ind[0])
            n_slice = slice(img_ind[1], end_n) \
                if end_n > img_ind[1] else \
                slice(end_n, img_ind[1])
            mask[m_slice, img_ind[1]] = 0
            mask[img_ind[0], n_slice] = 0
            # new_img[img_ind[0], img_ind[1], :] = (255, 0, 0)

    new_img[~mask] = 255
    # border is also funky, just trim it
    new_img[0, :] = 255
    new_img[-1, :] = 255
    new_img[:, 0] = 255
    new_img[:, -1] = 255
    # plt.subplot(121)
    # plt.imshow(new_img)
    # plt.subplot(122)
    # plt.imshow(new_img)
    # plt.show()
    return new_img, (m_start, n_start)


def fragment(img, m_min, m_max, n_min, n_max, frag_size=default_frag_size):
    if m_min > m_max - frag_size[0] or n_min > n_max - frag_size[1]:
        return None, (0, 0)

    # m_start = np.random.randint(m_min, m_max+1-frag_size)
    # n_start = np.random.randint(n_min, n_max+1-frag_size)

    # use normal dist, stdev range/2/2
    norm_scale = 2

    m_choices = np.arange(m_min, m_max+1-frag_size[0])
    n_choices = np.arange(n_min, n_max+1-frag_size[1])

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

    new_img = np.copy(img[m_start:m_start+frag_size[0], n_start:n_start+frag_size[1]])

    # random shapes method
    # shape = np.random.choice(['triangle', 'rectangle', 'circle'])
    # from skimage.draw import random_shapes
    # result, shapes = random_shapes(new_img.shape, max_shapes=1,
    #                                shape=shape, multichannel=False)
    # print(shapes)
    # mask = result != 255
    # new_img[~mask] = 255

    from skimage.draw import polygon

    # generate 4 random points along perimeter of new_img
    n_points = 4
    # perimeter = 2*(frag_size[0]+frag_size[1])
    perim_sizes = np.array([
        frag_size[0],
        frag_size[1],
        frag_size[0],
        frag_size[1],
    ])
    perim_sum = np.array([
        0,
        frag_size[0],
        frag_size[0]+frag_size[1],
        2*frag_size[0]+frag_size[1],
    ])
    offsets = np.random.random((n_points,))*perim_sizes
    offsets = offsets + perim_sum + np.random.random()*frag_size[0]
    def offset_to_xy(o):
        # clean way, way below is equivalent but branchless
        # if o < perim_sum[1]:
        #     return o, 0
        # elif o < perim_sum[2]:
        #     return frag_size[0], o-perim_sum[1]
        # elif o < perim_sum[3]:
        #     return frag_size[0] + (perim_sum[2]-o), frag_size[1]
        # else:
        #     return 0, frag_size[1] + (perim_sum[3]-o)

        # nvm this was hard
        # return np.array((o, 0)) * (0 < o < perim_sum[1]) * \
        #        np.array((frag_size[0], o-perim_sum[1])) * (perim_sum[1] < o < perim_sum[2]) * \
        #        np.array((frag_size[0]+perim_sum[2]-o, frag_size[1])) * (perim_sum[2] < o < perim_sum[3]) * \
        #        np.array((0, frag_size[1] + perim_sum[3]-o)) * (perim_sum[3] < o)
        # let me try splitting x and y

        x = o * (o < perim_sum[1]) + \
            frag_size[0] * np.logical_and(perim_sum[1] < o, o < perim_sum[2]) + \
            (frag_size[0] + perim_sum[2] - o)*np.logical_and(perim_sum[2] < o, o < perim_sum[3])

        y = (o-perim_sum[1]) * np.logical_and(perim_sum[1] < o, o < perim_sum[2]) + \
            frag_size[1]*np.logical_and(perim_sum[2] < o, o < perim_sum[3]) + \
            (frag_size[1] + perim_sum[3]-o)*(perim_sum[3] < o)

        return np.stack((x, y), axis=1)

    # poly = np.array([offset_to_xy(o) for o in offsets])
    poly = offset_to_xy(offsets)  # branchless
    mm, nn = polygon(poly[:, 0], poly[:, 1], new_img.shape)

    # offsets = perim_sizes / 2 + perim_sum
    # poly = np.array([offset_to_xy(o) for o in offsets])
    # poly = offset_to_xy(offsets)  # branchless
    # mm, nn = polygon(poly[:, 0], poly[:, 1], new_img.shape)

    # print(offsets)
    # print(poly)

    mask = np.zeros_like(new_img, dtype=bool)
    mask[mm, nn] = 1
    new_img[~mask] = 255
    # border is also funky, just trim it
    new_img[0, :] = 255
    new_img[-1, :] = 255
    new_img[:, 0] = 255
    new_img[:, -1] = 255
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(new_img)
    # plt.show()
    return new_img, (m_start, n_start)


def main_biggan():
    for f_img in glob.glob(dir_in + '/*'):
        img = dip.imread(f_img)
        img_id = int(os.path.split(f_img)[-1].split('.')[0])
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
            frag, _ = fragment(img, m_min, m_max, n_min, n_max)
            if frag is None:
                frag_failed = True
                break
            dip.im_write(frag, out_frag(img_id, n))

        if frag_failed:
            continue

        shutil.copyfile(f_img, out_img(img_id))

        # plt.close('all')
        # plt.subplot(221)
        # plt.imshow(img)
        #
        # plt.subplot(222)
        # plt.imshow(grad)
        #
        # plt.subplot(223)
        # plt.imshow(255*mask)
        #
        # plt.show()
        # plt.pause(1)


def main_pix2pix():
    def out_pix2pix_name():
        global _pix2pix_counter
        outname = f'{_pix2pix_dir}/{_pix2pix_counter}.jpg'
        _pix2pix_counter += 1
        return outname

    for f_img in glob.glob(dir_in + '/*'):
        try:
            img = dip.imread(f_img)
        except:
            continue
        img_out = dip.resize(img, _pix2pix_outsize, interpolation=INTER_AREA)
        print(f_img, img.shape)
        if len(img.shape) == 3:
            gray = np.mean(img, axis=-1)
        else:
            # gray = img
            continue
        grad = dip.transforms.edge_detect(gray)
        mask, m_min, m_max, n_min, n_max = space_fill(grad)

        for n in range(n_fragments):
            # frag_size = max(img.shape[0], img.shape[1]) // 4
            frag_size = img.shape[0]//4, img.shape[1]//4
            frag, _ = fragment(img, m_min, m_max, n_min, n_max, frag_size=frag_size)
            if frag is None:
                break
            else:
                frag = dip.resize(frag, _pix2pix_outsize, interpolation=INTER_AREA)
                both = np.concatenate([img_out, frag], axis=1)
                # plt.imshow(both)
                # plt.show()

                dip.im_write(both, out_pix2pix_name())


def isInHull(P,hull):
    '''
    Datermine if the list of points P lies inside the hull
    :return: list
    List of boolean where true means that the point is inside the convex hull
    '''
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all((A @ np.transpose(P)) <= np.tile(-b,(1,len(P))),axis=0)
    return isInHull


def main_pix2pix_context():
    def out_pix2pix_name():
        global _pix2pix_counter
        outname = f'{_pix2pix_dir}/{_pix2pix_counter}.jpg'
        _pix2pix_counter += 1
        return outname

    for f_img in tqdm(glob.glob(dir_in + '/*')):
        try:
            img = dip.imread(f_img)
        except:
            continue
        img_out = dip.resize(img, _pix2pix_outsize, interpolation=INTER_AREA)
        print(f_img, img.shape)
        if len(img.shape) == 3:
            gray = np.mean(img, axis=-1)
        else:
            # gray = img
            continue
        grad = dip.transforms.edge_detect(gray)
        mask, m_min, m_max, n_min, n_max = space_fill(grad)
        trimx = grad.shape[0] // 20
        trimy = grad.shape[1] // 20
        grad[:trimx, :] = 0
        grad[-trimx:, :] = 0
        grad[:, :trimy] = 0
        grad[:, -trimy:] = 0
        markerx = _pix2pix_marker_size * (grad.shape[0]) // _pix2pix_outsize[0] // 2
        markery = _pix2pix_marker_size * (grad.shape[1]) // _pix2pix_outsize[1] // 2

        grad_top = grad > np.percentile(grad, 99)
        border_inds = np.argwhere(grad_top)
        hull = ConvexHull(border_inds)
        perim = border_inds[hull.vertices]
        perim = np.concatenate((perim, border_inds[hull.vertices[:1]]), axis=0)
        _frag_context = np.zeros_like(grad_top)
        for a, b in zip(perim[:-1], perim[1:]):
            a, b = np.array(a), np.array(b)
            for t in np.linspace(0, 1, max(grad.shape)):
                p = t * a + (1 - t) * b
                ind = np.round(p).astype(np.int)
                mstart = max(ind[0] - markerx, 0)
                mstop = min(ind[0] + markerx, grad.shape[0] - 1)
                nstart = max(ind[1] - markery, 0)
                nstop = min(ind[1] + markery, grad.shape[1] - 1)
                _frag_context[mstart:mstop, nstart:nstop] = 1
        mm = list(np.ndindex(_frag_context.shape[:2]))
        out_hull = isInHull(mm, hull)
        mm = np.array(mm)[~out_hull]
        # print(mm)
        # input(out_hull)
        _frag_context = np.stack([255*~_frag_context]*3, axis=-1).astype(np.uint8)
        _frag_context[mm[:, 0], mm[:, 1]] = img[mm[:, 0], mm[:, 1]]

        # binary erosion - erodes binary blocks like sand bars
        # frag_context2 = frag_context & binary_erosion(frag_context)

        # threshold/contour method
        # ret, thresh = cv2.threshold(grad, np.percentile(grad, 95), 255, 0)
        # contours, hier = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours, hier.shape)
        # frag_context = np.zeros_like(grad)
        # cv2.drawContours(frag_context, contours, -1, 255, 3)
        # draw contours according to hierarchy
        # frag_context2 = np.zeros_like(grad)
        # hier = (hier - np.min(hier)) / (np.max(hier) - np.min(hier))
        # for h, contlist in zip(hier[0], contours):
        #     for cont in contlist:
        #         frag_context2[cont[0, 1], cont[0, 0]] = max(h)

        # otsu thresholding
        # val = filters.threshold_otsu(gray); frag_context = gray < val
        for n in range(n_fragments):
            # frag_size = max(img.shape[0], img.shape[1]) // 4
            frag_size = img.shape[0]//4, img.shape[1]//4
            frag, frag_pos = fragment(img, m_min, m_max, n_min, n_max, frag_size=frag_size)
            fragx = slice(frag_pos[0], frag_pos[0]+frag_size[0])
            fragy = slice(frag_pos[1], frag_pos[1]+frag_size[1])
            if frag is None:
                break
            else:
                frag_context = np.copy(_frag_context)
                white = (255, 255, 255)
                frag_context[fragx, fragy][frag != white] = frag[frag != white]
                # img = np.copy(img)
                # img[fragx, fragy][frag != white] = frag[frag != white]
                frag_context_out = dip.resize(frag_context, _pix2pix_outsize, interpolation=INTER_AREA)

                # frag_context_out = frag_context_out[:, :, 0]
                # print(img_out.shape, frag_context_out.shape)
                # print(img_out.dtype, frag_context_out.dtype)
                both = np.concatenate([img_out, frag_context_out], axis=1)

                # plt.subplot(221)
                # plt.imshow(img)
                # plt.subplot(222)
                # plt.imshow(grad)
                # plt.subplot(223)
                # plt.imshow(frag_context)
                # plt.subplot(224)
                # plt.imshow(both)
                # plt.show()

                dip.im_write(both, out_pix2pix_name())
                # break

if __name__ == '__main__':
    main_pix2pix_context()
    # main_pix2pix()
    # main_biggan()

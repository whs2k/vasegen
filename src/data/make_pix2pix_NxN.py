import os
import sys
import glob
from tqdm import tqdm
import dippykit as dip
from cv2 import INTER_AREA, INTER_CUBIC


try:
    outsize = (int(sys.argv[1])*2, int(sys.argv[1]))
except:
    print('provide a number for the side of the square image to scale to')
    exit()

_pix2pix_indir = 'data/processed/pix2pix_vase_fragments/train/'
_pix2pix_outdir = f'data/processed/pix2pix_vase_fragments_{sys.argv[1]}/train/'
out_pix2pix = lambda fname: f'{_pix2pix_outdir}/{fname}.jpg'

if os.path.exists(_pix2pix_outdir):
    y_n = input(f'Folder {_pix2pix_outdir} exists, overwrite?')
    if 'y' not in y_n:
        exit()


os.makedirs(_pix2pix_outdir, exist_ok=True)

for f_img in tqdm(glob.glob(_pix2pix_indir + '/*.jpg')):
    try:
        img = dip.imread(f_img)
    except:
        continue

    img_name = os.path.split(f_img)[-1]

    img = dip.resize(img, outsize, interpolation=INTER_CUBIC)
    dip.im_write(img, out_pix2pix(img_name))
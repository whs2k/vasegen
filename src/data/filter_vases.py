import os
import pickle
import shutil
import dippykit as dip


info_fname = 'data/raw/vase_info.pkl'

id_to_img_name = lambda _img_id: f'data/raw/vase_imgs/{_img_id}.jpg'
id_to_out_name = lambda _img_id: f'data/processed/full_vases/{_img_id}.jpg'

outsize = (512, 512)

with open(info_fname, 'rb') as f:
    all_info = pickle.load(f)

print(len(all_info), 'vases')
for img_id, img_info in all_info.items():
    if 'Fragments' in img_info['categories'] \
            or 'fragment' in img_info['description'].lower():
        continue
    else:
        try:
            img = dip.imread(id_to_img_name(img_id))
        except:
            continue
        print(img.shape)
        new_img = dip.resize(img, dsize=outsize)
        dip.im_write(new_img, id_to_out_name(img_id))


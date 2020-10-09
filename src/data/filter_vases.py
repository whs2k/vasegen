import os
import shutil
import pickle
import dippykit as dip


info_fname = 'data/raw/vase_info.pkl'

dir_out = 'data/processed/full_vases/'
id_to_img_name = lambda _img_id: f'data/raw/vase_imgs/{_img_id}.jpg'
id_to_out_name = lambda _img_id: f'{dir_out}/{_img_id}.jpg'

default_outsize = (512, 512)

if not os.path.exists(dir_out):
    os.mkdir(dir_out)

with open(info_fname, 'rb') as f:
    all_info = pickle.load(f)

def main(outsize=default_outsize):
    print(len(all_info), 'vases')
    for img_id, img_info in all_info.items():
        # input(img_info)  # if you want to see one line
        if 'fragment' in img_info['Title'].lower() or \
                'fragment' in img_info['description'].lower() or \
                'Fragments' in img_info['categories']:
            continue
        # just get Terracotta
        elif 'terra' in img_info['Title'].lower() or \
                'Medium' not in img_info or \
                ('Medium' in img_info and 'terra' in img_info['Medium'].lower()) or \
                'terra' in img_info['description'].lower() or \
                'Terracotta' in img_info['categories']:
        # just get everything that's not broken
        # else:
            if outsize:
                try:
                    img = dip.imread(id_to_img_name(img_id))
                except:
                    continue
                print(img.shape)
                new_img = dip.resize(img, dsize=outsize)
                dip.im_write(new_img, id_to_out_name(img_id))
            else:
                shutil.copyfile(id_to_img_name(img_id), id_to_out_name(img_id))


if __name__ == '__main__':
    # main((512, 512))  # do resize all
    main(None)  # don't resize all


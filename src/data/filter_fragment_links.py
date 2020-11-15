import os
import shutil
import pickle
import dippykit as dip
from cv2 import INTER_AREA
from tqdm import tqdm

info_fname = 'data/raw/vase_info.pkl'

with open(info_fname, 'rb') as f:
    all_info = pickle.load(f)

dir_out = 'data/processed/fragments/'
id_to_img_name = lambda _img_id: f'data/raw/vase_imgs/{_img_id}.jpg'
id_to_out_name = lambda _img_id: f'{dir_out}/{_img_id}.jpg'

outsize = (512, 512)

if not os.path.exists(dir_out):
    os.mkdir(dir_out)

def main():
    print(len(all_info), 'vases')
    link_dict = dict()
    for img_id, img_info in tqdm(all_info.items()):
        # input(img_info)  # if you want to see one line
        if 'fragment' in img_info['Title'].lower() or \
                'fragment' in img_info['description'].lower() or \
                'Fragments' in img_info['categories']:
            link_dict[img_id] = img_info['src']
            try:
                img = dip.imread(id_to_img_name(img_id))
            except:
                continue
            print(img.shape)
            new_img = dip.resize(img, dsize=outsize, interpolation=INTER_AREA)
            dip.im_write(new_img, id_to_out_name(img_id))
        elif 'terra' in img_info['Title'].lower() or \
                'Medium' not in img_info or \
                ('Medium' in img_info and 'terra' in img_info['Medium'].lower()) or \
                'terra' in img_info['description'].lower() or \
                'Terracotta' in img_info['categories']:
            pass
            # if outsize:
            #     try:
            #         img = dip.imread(id_to_img_name(img_id))
            #     except:
            #         continue
            #     print(img.shape)
            #     new_img = dip.resize(img, dsize=outsize, interpolation=INTER_AREA)
            #     dip.im_write(new_img, id_to_out_name(img_id))
            # else:
            #     shutil.copyfile(id_to_img_name(img_id), id_to_out_name(img_id))
    with open('data/raw/frag_links.pkl', 'wb') as f:
        pickle.dump(link_dict, f)


if __name__ == '__main__':
    main()

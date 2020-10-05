import os
import time
import pickle
import shutil
import requests
from requests_html import HTMLSession

from threading import Lock, Thread

from tqdm import tqdm

vase_fname = 'data/raw/vase_links_all.txt'
info_fname = 'data/raw/vase_info.pkl'
n_vases = 25610  # may change if file above is regenerated

def requests_setup():
    session = HTMLSession()
    return session

def download_img(url, fname):
    img_data = requests.get(url, stream=True)
    img_data.raw.decode_content = True
    with open(fname, 'wb') as img_f:
        shutil.copyfileobj(img_data.raw, img_f)
    del img_data

def get_img_info_thread(url):
    img_id = int(url.split('/')[-1])
    # uncomment this and download_img if you want it all in one loop
    # I don't do that to slow down requests a bit
    # img_fname = f'data/raw/vase_imgs/{img_id}.jpg'
    # if os.path.exists(img_fname):
    #     return

    # split getting info and downloading image requests
    # cut bombarding server in half
    # also cap the number of concurrent threads in __main__
    if img_id not in all_info:
        session = requests_setup()
        r = session.get(url)

        imgs = r.html.find('.artwork__image')
        # make sure there's just one so there's no ambiguity
        try:
            assert len(imgs) == 1
        except AssertionError:
            return

        src = imgs[0].attrs['src']

        # download_img(src, img_fname)

        info_section = r.html.find('.artwork-info')[0]
        keys = info_section.find('.artwork__tombstone--label')
        vals = info_section.find('.artwork__tombstone--value')
        keys = [key.text[:-1] for key in keys]
        vals = [val.text for val in vals]

        img_info = dict(zip(keys, vals))
        img_info['src'] = src

        facets = r.html.find('.artwork__facets')

        # also get categories
        categories = facets[1].find('a')
        categories = [c.text.split()[0] for c in categories]

        # locations
        # locations = facets[2].find('a')
        # locations = [l.text.split()[0] for l in locations]

        # date/era
        # era = facets[3].find('a')[0]
        # era = era.text.split()[0]


        img_info['categories'] = categories
        # img_info['location'] = location
        # img_info['era'] = era

        with all_info_lock:
            all_info[img_id] = img_info
    else:
        pass
        # print('skipping populated ID', img_id)


def get_img_thread(url):
    img_id = int(url.split('/')[-1])
    img_fname = f'data/raw/vase_imgs/{img_id}.jpg'
    if os.path.exists(img_fname):
        return

    if img_id in all_info:
        src = all_info[img_id]['src']
        download_img(src, img_fname)
    else:
        pass
        # print('skipping unpopulated ID', img_id)


if __name__ == '__main__':
    if os.path.exists(info_fname):
        with open(info_fname, 'rb') as f:
            all_info = pickle.load(f)
    else:
        all_info = dict()

    all_info_lock = Lock()
    threads = list()
    thread_targets = list()
    thread_targets.append([get_img_info_thread, False, 32])
    thread_targets.append([get_img_thread, True, 10])
    for thread_target, do_run, max_threads in thread_targets:
        if do_run:
            with open(vase_fname, 'r') as f_links:
                for line in tqdm(f_links, total=n_vases):
                    url = line.strip()  # trim newline
                    # get_img_thread(url)  # for testing
                    t = Thread(target=thread_target, args=(url,))
                    t.start()
                    threads.append(t)
                    if len(threads) >= max_threads:
                        for t in threads:
                            t.join()
                        threads = []

            for t in threads:
                t.join()

            if thread_target == get_img_info_thread:
                with open(info_fname, 'wb') as f:
                    pickle.dump(all_info, f)


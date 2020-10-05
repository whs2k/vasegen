import os
import re
import time
import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession,AsyncHTMLSession

from selenium import webdriver

search_url = 'https://www.metmuseum.org/art/collection/search#!?material=Vases'
search_res_cls = 'result_card__link'
search_link_re = r'search/\d{1,7}\?'
vase_fname = 'data/raw/vase_links_all.txt'

async def requests_setup():
    asession = AsyncHTMLSession()
    return asession

async def requests_get(asession, _url):
    r = await asession.get(_url)

    await r.html.arender()
    return r

def wait_load(_t=3):
    time.sleep(_t)

def sel_setup():
    browser = webdriver.Chrome()
    return browser

def sel_get(browser, _url):
    return browser.get(_url)

def get_search_links(browser):
    _links = browser.find_elements_by_css_selector('a')
    ## TODO: this following line fails sometimes, StaleElementReferenceException
    # upping the wait time to 10 seconds seems to have fixed. Maybe 5s is enough
    _links = [l.get_property('href') for l in _links]
    _matches = [re.search(search_link_re, l) for l in _links]
    _search_links = [link[:match.span()[1]-1]
                     for link, match in zip(_links, _matches)
                     if match]
    return _search_links

def next_search_page(browser):
    next_page_cls = 'pagenav-numeric__next'
    next_page_button = browser.find_elements_by_class_name(next_page_cls)
    assert len(next_page_button) == 1
    next_page_button[0].click()


if __name__ == '__main__':
    if os.path.exists(vase_fname):
        res = input(vase_fname + ' already exists, do you want to overwrite? [y or n]')
        if 'y' in res:
            print('Erasing ' + vase_fname)
            with open(vase_fname, 'w') as f_links:
                pass
        else:
            print('Exiting')
            exit(0)

    browser = sel_setup()
    sel_get(browser, search_url)
    wait_load()  # let javascript load

    links = browser.find_elements_by_css_selector('a')
    eighty_per_page = [l for l in links if l.text == '80']
    assert len(eighty_per_page) == 1
    eighty_per_page[0].click()
    wait_load() # let new page load

    search_links = set()
    while True:
        print(f'Currently have {len(search_links)} links')
        # links = browser.find_elements_by_class_name(search_res_cls)
        # messy, but try twice then break to salvage search_links object
        new_links = get_search_links(browser)
        search_links.update(new_links)
        with open(vase_fname, 'a') as f_links:
            for link in new_links:
                f_links.write(link)
                f_links.write('\n')
        try:
            next_search_page(browser)
            wait_load(10)
        except AssertionError:
            break

    # print(f'writing out {len(search_links)} search links')
    # with open(vase_fname, 'w') as f_links:
    #     for link in search_links:
    #         f_links.write(link)
    #         f_links.write('\n')

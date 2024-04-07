# -*- encoding: utf-8 -*-
# @File        :   crawler.py
# @Time        :   2024/03/28 15:25:18
# @Author      :   Siyou
# @Description :

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from collections import OrderedDict
from tqdm import tqdm

page = requests.get('https://www.goodreads.com/quotes')
if page.status_code == 200:
    pageParsed = BeautifulSoup(page.content, 'html5lib')
    
# Define a function that retrieves information about each HTML quote code in a dictionary form.
def extract_data_quote(quote_html):
        quote = quote_html.find('div',{'class':'quoteText'}).get_text().strip().split('\n')[0]
        author = quote_html.find('span',{'class':'authorOrTitle'}).get_text().strip()
        if quote_html.find('div',{'class':'greyText smallText left'}) is not None:
            tags_list = [tag.get_text() for tag in quote_html.find('div',{'class':'greyText smallText left'}).find_all('a')]
            tags = list(OrderedDict.fromkeys(tags_list))
            if 'attributed-no-source' in tags:
                tags.remove('attributed-no-source')
        else:
            tags = None
        data = {'quote':quote, 'author':author, 'tags':tags}
        return data

# Define a function that retrieves all the quotes on a single page. 
def get_quotes_data(page_url):
    page = requests.get(page_url)
    if page.status_code == 200:
        pageParsed = BeautifulSoup(page.content, 'html5lib')
        quotes_html_page = pageParsed.find_all('div',{'class':'quoteDetails'})
        return [extract_data_quote(quote_html) for quote_html in quotes_html_page]

# Retrieve data from the first page.
data = get_quotes_data('https://www.goodreads.com/quotes')

# Retrieve data from all pages.
for i in tqdm(range(2,1000)):
    url = f'https://www.goodreads.com/quotes?page={i}'
    data_current_page = get_quotes_data(url)
    if (data_current_page is None) or (data_current_page in data):
        continue
    data = data + data_current_page

data_df = pd.DataFrame.from_dict(data)
for i, row in data_df.iterrows():
    if row['tags'] is None:
        data_df = data_df.drop(i)
# Produce the data in a JSON format.
data_df.to_json('res/quotes.jsonl',orient="records", lines =True,force_ascii=False)
# Then I used the familiar process to push it to the Hugging Face hub.

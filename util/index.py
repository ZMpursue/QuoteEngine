# -*- encoding: utf-8 -*-
# @File        :   index.py
# @Time        :   2024/03/28 16:08:41
# @Author      :   Siyou
# @Description :

from embedding_model.multilingualCLIP import MCLIP
from embedding_model.multilingualMiniLM import MiniLM
import json
from tqdm import tqdm
import nmslib
import pandas as pd

emb_model = MiniLM()
with open("res/quotes.csv", encoding="utf8") as file:
    data_csv = pd.read_csv(file)
index = nmslib.init(method="hnsw", space="cosinesimil")
print("Index initialized")

quote_id = 0
max_quote_id = len(data_csv)
for i, row in tqdm(data_csv.iterrows()):
    quote_id += 1
    text = row["quote"]
    try:
        embeddings = emb_model.inference(sentences=text)
    except Exception as e:
        print(e)
    index.addDataPoint(id=quote_id, data=embeddings)
    if quote_id >= max_quote_id:
        break
index.createIndex({"M": 16, "efConstruction": 12, "post": 2}, print_progress=True)
index.saveIndex("res/index.bin")

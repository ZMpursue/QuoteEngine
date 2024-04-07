# -*- encoding: utf-8 -*-
# @File        :   web_service.py
# @Time        :   2024/03/28 18:29:06
# @Author      :   Siyou
# @Description :

import json
import os
import shutil
from typing import Optional, Awaitable

import numpy as np
import tornado.ioloop
import tornado.web  
import nmslib
from embedding_model.multilingualCLIP import MCLIP
from embedding_model.multilingualMiniLM import MiniLM
import pandas as pd
import tqdm

model = MiniLM()

# load quotes
quotes = {}
quotes_id = 0
with open("res/quotes.csv", encoding="utf8") as file:
    data_csv = pd.read_csv(file)
max_quote_id = 10000
for i, row in data_csv.iterrows():
    quotes_id += 1
    quotes[str(quotes_id)] = {"quote":row["quote"], "author":row["author"], "category":row["category"]}
print("[*] Quotes loaded...")

# load index
index_ = nmslib.init(method="hnsw", space="l2")
index_.loadIndex("res/index.old.bin", load_data=False)
print("[*] Index loaded...")

print("[*] Web service started...")
print("[*] Please visit http://0.0.0.0:9999/QuoteEngine")
class FormsHandler(tornado.web.RequestHandler):
    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass

    def get(self):
        user_quote = ""
        user_images = []
        match = {"data": []}
        try:
            with open("./web/static/cache/activate.json", encoding="utf8") as file:
                activate = json.loads(file.read())
            user_quote = activate["user_quote"]
            user_images = activate["user_images"]

            with open("./web/static/cache/match.json", encoding="utf8") as file:
                match = json.loads(file.read())
        except Exception as e:
            print(e)
        self.render('index.html', 
                    user_quote=user_quote, 
                    user_images=user_images,
                    match=match["data"],
                    )

    def post(self):
        if not os.path.exists("./web/static/cache"):
            os.mkdir("./web/static/cache")
        # delete old match and activate cache
        try:
            shutil.rmtree("./web/static/cache/activate.json")
        except Exception as e:
            pass
        try:
            shutil.rmtree("./web/static/cache/match.json")
        except Exception as e:
            pass
        user_images_path = "./web/static/cache/image" 

        file_metas = self.request.files.get('image', [])  
        user_quote = self.get_argument("user_quote")
        images_paths = []
        for meta in file_metas:
            file_name = meta.get('filename')
            file_path = os.path.join(user_images_path, file_name)  
            images_paths.append(file_path)
            with open(file_path, 'wb') as f:
                f.write(meta.get('body'))  

        activate = {"user_quote": user_quote, "user_images": images_paths}
        with open("./web/static/cache/activate.json", "w", encoding='utf8') as file:
            json_data = json.dumps(activate, ensure_ascii=False)
            file.write(json_data)
            file.close()

        try:
            embeddings = model.inference(sentences=user_quote)
            res = index_.knnQuery(embeddings, k=6)
            print("[*]Query result: \n\tindex: ", res[0].tolist(), "\n\tdistances: ", res[1].tolist())
            match = []
            for i in range(len(res[0])):
                var_data = quotes[str(res[0][i])]
                var_data["confidence"] = str(res[1][i])
                match.append(var_data)
            match_ = {"data": match}
            with open("./web/static/cache/match.json", "w", encoding='utf8') as file:
                json_data = json.dumps(match_, ensure_ascii=False)
                file.write(json_data)
                file.close()
        except Exception as e:
            print(e)
        return self.redirect('/QuoteEngine')


def make_app():
    setting = dict(
        template_path=os.path.join(os.path.dirname(__file__), "./template"),
        static_path=os.path.join(os.path.dirname(__file__), "./static"),
    )
    return tornado.web.Application([  
        (r"/QuoteEngine", FormsHandler), 
    ],
        **setting
    )


if __name__ == "__main__":  
    app = make_app()
    app.listen(9999)
    tornado.ioloop.IOLoop.current().start() 

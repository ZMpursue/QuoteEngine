# -*- encoding: utf-8 -*-
# @File        :   multilingualCLIP.py
# @Time        :   2024/03/28 15:50:21
# @Author      :   Siyou
# @Description :

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MCLIP:
    def __init__(self, device: str = "mps"):
        self.device = device
        model = SentenceTransformer(
            "sentence-transformers/clip-ViT-B-32-multilingual-v1",
            device=self.device)
        image_model = SentenceTransformer("clip-ViT-B-32",
                                          device=self.device)
        self.model = model
        self.image_model = image_model

    def inference(self, sentences: str, images: list) -> list:
        if len(images) != 0:
            img = Image.open(images[0])
            img = img.convert('RGB')
            images_embeddings = self.image_model.encode([img])
            # adopt average pooling
            images_embeddings = np.mean(images_embeddings, 0)
            return images_embeddings.tolist()

        elif sentences != "":
            sentences_embeddings = self.model.encode([sentences])[0]
            return sentences_embeddings.tolist()
        else:
            print("ERROR: No input data")

if __name__ == "__main__":
    emb_model = MCLIP()
    emb = emb_model.inference("Hello, world!", [])
    print(len(emb))

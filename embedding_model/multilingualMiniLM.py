# -*- encoding: utf-8 -*-
# @File        :   multilingualMiniLM.py
# @Time        :   2024/03/31 00:34:42
# @Author      :   Siyou
# @Description :

from transformers import AutoTokenizer, AutoModel
import torch

class MiniLM:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(self.device)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def inference(self, sentences: str) -> list:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])[0].tolist()
        return sentence_embeddings
    
if __name__ == "__main__":
    emb_model = MiniLM()
    emb = emb_model.inference("Hello, world!")
    print(len(emb))
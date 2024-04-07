![](./web/static/images/logo.svg)
> A demo of information retrieval system.
> This project is base on sentence embedding and HNSW algorithm. Using the sentence embedding to convert the sentence into a vector, and using the HNSW algorithm to build the index for the vectors. The system can retrieve the most similar sentence to the query sentence.

## Features
- Sentence embedding: This is a sentence-transformers model. It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search. more details can be found [here](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- HNSW algorithm: This is a fast approximate nearest neighbor search algorithm which is used to build the index for the sentence embedding vectors and can be used to retrieve the most similar sentence to the query sentence. more details can be found [here](https://arxiv.org/pdf/1603.09320)
## Prerequisites
- Python 3.6+
- tornado==6.4
- torchvision==0.15.2
- torch==2.1.2

## Download the pre-trained model and data
### 1. download the Quora-500k dataset
```bash
cd res
wget https://huggingface.co/datasets/jstet/quotes-500k/resolve/main/quotes.csv\?download\=true
```
## Usage
### 1. generate the sentence embedding index
```bash
python3 -m util.index
```
### 2. start the web server
```bash
python -m web.web_service 
```
### 3. open the browser and visit the web page
[http://0.0.0.0:9999/QuoteEngine](http://0.0.0.0:9999/QuoteEngine)
![](./web/static/images/web_page.png)
## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


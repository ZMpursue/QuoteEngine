# -*- encoding: utf-8 -*-
# @File        :   HNSW.py
# @Time        :   2024/03/28 18:30:35
# @Author      :   Siyou
# @Description :

import nmslib

class HNSW:
    def __init__(self):
        self.hnsw = nmslib.init(method="hnsw", space="cosin")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim 
from gensim import models
from sklearn.ensemble import IsolationForest
from collections import defaultdict

model = models.Doc2Vec.load('D:\model\model')

# モデルの中から、購入アイテムがmin_doc_count以上のdoctagのリストを取得する
def get_target_doc_list(min_doc_count):
  all_doc_list = []
  for row in model.docvecs.doctags:
    all_doc_list.append(row)

  # 全docタグのmapを作成
  all_doc_dict = defaultdict(lambda: list())
  for doc_id in all_doc_list:
    key = doc_id.split("_")[0]
    all_doc_dict[key].append(doc_id)

  # 要素数min_doc_count以上のdoc_idのリストを作成
  target_doc_list = []
  for doc_list in all_doc_dict.values():
    if (len(doc_list) >= min_doc_count):
      target_doc_list.append(doc_list)
  return target_doc_list

# ノイズとなるdoctagのリストをsvnで取得する。
def get_noise_docs(min_doc_count):
  noise_doc=[]
  for item_vecs in get_target_doc_list(min_doc_count):
    vecs = [model.docvecs[doc_key] for doc_key in item_vecs]
    #外れ値検出
    clf = IsolationForest(n_estimators=100,contamination=0.15)
    clf.fit(vecs)
    svm_results = clf.predict(vecs)

    for doc_key,result in zip(item_vecs,svm_results):
      if result == -1:
        noise_doc.append(doc_key)
        print(doc_key)

  return noise_doc

# file = open('noise_doc_tag.txt','w')
# file.write(','.join(get_noise_docs(5)))
# file.close()


file = open('all_doc_tag.txt','w')
for docs in get_target_doc_list(100):
  file.write(','.join(docs)+'\n')

file.close()
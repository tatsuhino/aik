#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim 
from gensim import models
from sklearn.ensemble import IsolationForest

model = models.Doc2Vec.load('D:\model\model')

# 購入された数の多いアイテムについて、履歴ベクトルをファイルから読み込み
def get_top_item_docs():
  top_item=[]
  with open('plot.txt', 'r') as f:
    for line in f:
      top_item.append(line.replace('\n','').split(","))
      
  return top_item


def draw_word_scatter():

  # matplotlibによる可視化
  # Scikit-learnのPCAによる次元削減とその可視化
  pca = PCA(n_components=2)
  all_color = ["#ff0000","#ff00ff","#7f00ff","#0000ff","#007fff","#00ffff","#00ff7f","#00ff00","#7fff00","#ff7f00","#ff7f7f","#ff7fbf","#ff7fff","#bf7fff","#7f7fff","#7fbfff","#7fffff","#7fffbf","#7fff7f","#bfff7f","#ffbf7f"]
  # fig, ax = plt.subplots() # 一枚のグラフにする場合

  for i,item_vecs in enumerate(get_top_item_docs()):
    

    vecs = []
    for doc_key in item_vecs:
      if model.docvecs.doctags[doc_key].word_count > 3:
        vecs.append(model.docvecs[doc_key])

    # vecs = [model.docvecs[doc_key] for doc_key in item_vecs]

    #外れ値検出
    clf = IsolationForest(n_estimators=50,contamination=0.15)
    clf.fit(vecs)
    svm_result = clf.predict(vecs)

    # グラフ描画
    coords = pca.fit_transform(vecs)
    color = all_color[i]
    fig, ax = plt.subplots()
    x = [v[0] for v in coords]
    y = [v[1] for v in coords]
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    for j, txt in enumerate(vecs):
      if svm_result[j] == 1:
        ax.scatter(x[j], y[j], lw=0.1,c="#00ffff")
        # ax.scatter(x[j], y[j], lw=0.1,c=color) # 色変えたい場合はこっち
      else:
        ax.scatter(x[j], y[j], lw=0.1,c="red")
      
      # ax.annotate(txt.split("_")[0], (coords[i][0], coords[i][1])) #プロットにラベルつける場合

  plt.show()


draw_word_scatter()
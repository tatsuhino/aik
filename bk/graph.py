#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gensim 
from gensim import models

model = models.Doc2Vec.load('D:\model\model')

def draw_word_scatter(doc, topn=30):
  """ 入力されたwordに似ている単語の分布図を描くためのメソッド """
  
  doc_keys = [x[0] for x in sorted(model.docvecs.most_similar(doc, topn=topn))]
  vecs = [model.docvecs[doc_key] for doc_key in doc_keys]

  # 分布図
  draw_scatter_plot(vecs, doc_keys)


def draw_scatter_plot(vecs, tags):
  """ 入力されたベクトルに基づき散布図(ラベル付き)を描くためのメソッド """

  # Scikit-learnのPCAによる次元削減とその可視化
  pca = PCA(n_components=2)
  coords = pca.fit_transform(vecs)

  # matplotlibによる可視化
  fig, ax = plt.subplots()
  x = [v[0] for v in coords]
  y = [v[1] for v in coords]

  # ax.scatter(x, y)
  ax.set_xlim(-1,1)
  ax.set_ylim(-1,1)

  for i, txt in enumerate(tags):
    ax.scatter(x[i], y[i], c='red')
    ax.annotate(txt.split("_")[0], (coords[i][0], coords[i][1])) 
    # ax.annotate(coords[i][0], coords[i][1])
  plt.show()

draw_word_scatter('320130_52316', topn=20)
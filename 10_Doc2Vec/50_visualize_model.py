#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    DOC2VECで学習したモデルを可視化
"""

# 共通
import time
import argparse
from logging import StreamHandler, Formatter, INFO,getLogger
import codecs
import collections
# Doc2Vec用ライブラリ
import gensim 
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# グラフ描画用ライブラリ
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 定数
BASE_DIR = "./5_Doc2Vec_サンプル"

# グローバル変数
logger = getLogger(__name__)

# 実行時引数
parser = argparse.ArgumentParser()
# Doc2Vec学習パラメータ
parser.add_argument('-vector_size', default=100, type=int) # ベクトルの次元数
parser.add_argument('-min_count', default=2, type=int) # 出現数がmin_count以下のアイテムIDは無視する
parser.add_argument('-epochs', default=55, type=int) # 一つの訓練データを何回繰り返して学習させるか(多すぎると過学習となる)
args = parser.parse_args()

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

# 学習
def train(train_data):
    train_corpus = [TaggedDocument(words=data_dict["view_items"], tags=[data_dict["tag_name"]]) for data_dict in train_data]
    model = Doc2Vec(vector_size=args.vector_size, min_count=args.min_count, epochs=args.epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    return model

# 履歴ファイルの読み込み
def read_history_data(file):
    all_data = []
    with open(file, mode='r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            data_dict={}
            history =  line.split(",")[2].rstrip()
            data_dict["view_items"] = history.rstrip().split(" ")[:-1] # TODO 最初にアイテムを見るまで
            data_dict["buy_item"] = history.split(" ")[-1].rstrip()
            data_dict["tag_name"] = str(i) + "_" + data_dict["buy_item"]
            all_data.append(data_dict)
            i += 1
    return all_data

def draw_word_scatter(model,top_seller_item_list):
    # matplotlibによる可視化
    pca = PCA(n_components=2)
    for top_seller_item in top_seller_item_list:
        # 閲覧履歴が3以上
        vecs = []
        for doc_tag in top_seller_item:
            # if model.docvecs.doctags[doc_tag].word_count > 4:
            vecs.append(model.docvecs[doc_tag])
        # グラフ描画
        coords = pca.fit_transform(vecs)
        fig, ax = plt.subplots()
        x = [v[0] for v in coords]
        y = [v[1] for v in coords]
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)

        for j, txt in enumerate(vecs): ax.scatter(x[j], y[j], lw=0.1,c="red")

    plt.show()

def main():
    # データ読み込み
    all_data = read_history_data(BASE_DIR + './events.csv_converted')
    model = train(all_data)
    logger.info("[モデル学習完了]")
    all_buy_item = [line_dict["buy_item"] for line_dict in all_data ]
    top_seller = [common[0] for common in collections.Counter(all_buy_item).most_common()[:10]]

    top_seller_data_list=[]
    for top_item_id in top_seller:
        # 以下では抽出できない？
        # history = list(filter(lambda x: x["tag_name"].endswith(top_item_id) , all_data))
        history = []
        for data in all_data:
            if data["tag_name"].endswith(top_item_id):
                history.append(data["tag_name"])
        top_seller_data_list.append(history)

    # 可視化
    draw_word_scatter(model,top_seller_data_list)

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")

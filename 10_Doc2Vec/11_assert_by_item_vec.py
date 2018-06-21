#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    DOC2VECで学習＋評価。アイテムをベクトル化し、ユーザが購入する直前のアイテムに似ているアイテムを表示する手法
"""

# 共通
import time
import argparse
from logging import StreamHandler, Formatter, INFO,getLogger
import codecs
# Doc2Vec用ライブラリ
import gensim 
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# 交差検証用ライブラリ
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 定数
BASE_DIR = "./10_Doc2Vec"

# グローバル変数
logger = getLogger(__name__)

# 実行時引数
parser = argparse.ArgumentParser()
# Doc2Vec学習パラメータ
parser.add_argument('-vector_size', default=200, type=int) # ベクトルの次元数
parser.add_argument('-min_count', default=2, type=int) # 出現数がmin_count以下のアイテムIDは無視する
parser.add_argument('-epochs', default=50, type=int) # 一つの訓練データを何回繰り返して学習させるか(多すぎると過学習となる)
args = parser.parse_args()

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

# おすすめアイテムの上位N件を取得(購入直前のアイテム２つに対して、類似のアイテムを取得)
def get_predict_item_top_n(model,test_data_line_dict, n):
    except_buy_item_history = list(filter(lambda item: item != test_data_line_dict["buy_item"], test_data_line_dict["view_items"]))
    # 購入の直前に見たアイテムから操作する
    except_buy_item_history.reverse()

    predict_dict = {}
    predict_dict["exceed_sim_vecs"] = []
    predict_dict["sim_vecs"] = []
    for item in except_buy_item_history :
        try:
            predict_item = model.most_similar(positive=str(item),topn=int(n))
            predict_dict["exceed_sim_vecs"].extend(predict_item)
            predict_dict["sim_vecs"].extend(predict_item[:int(n/2 + n%2)]) # TODO 直近２件決め打ちなロジック
            if len(predict_dict["exceed_sim_vecs"]) >= (n*2): break
        except KeyError: continue # ボキャブラリーに該当itemが存在しない場合
    
    # 予想アイテム数がnを超える前にループを抜けた場合は、exceed_sim_vecsから持ってくる
    if len(predict_dict["sim_vecs"]) < n:
        predict_dict["sim_vecs"].extend(predict_dict["exceed_sim_vecs"])
    if len(predict_dict["sim_vecs"]) == 0 : return [] 

    all_predict_item = [vec[0] for vec in predict_dict["sim_vecs"]]
    return list(set(all_predict_item))[:n]

# 検証
def is_hit(model,test_data_line_dict):
    predict_item = get_predict_item_top_n(model,test_data_line_dict,10)
    if test_data_line_dict["buy_item"] in predict_item: return True

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

def main():
    # データ読み込み
    all_data = read_history_data(BASE_DIR + './events.csv_converted')
    # 交差検証準備
    rule = KFold(n_splits=10, shuffle=True, random_state=1)
    all_data_frame = pd.DataFrame(all_data)

    hit_count_all=0
    for train_index,test_index in rule.split(all_data_frame):
        logger.info(">>>>>>[交差検証開始]")
        # 学習
        train_data = [all_data[i] for i in train_index]
        test_data = [all_data[i] for i in test_index]
        model = train(train_data)
        logger.info(">>>[モデル学習完了]")
        # 評価
        hit_count = 0
        for test_data_line in test_data:
            if is_hit(model,test_data_line) : hit_count += 1
        # 評価の印字
        logger.info(">>>[評価履歴数]" + str(len(test_data)))
        logger.info(">>>[ヒット数]" + str(hit_count))
        logger.info(">>>ヒット率]" + str(hit_count / len(test_data)))
        hit_count_all += hit_count
        logger.info(">>>>>>[交差検証終了]")
        break
    
    logger.info("------------------------------------------------")
    # 総合評価
    logger.info("[評価履歴数]" + str(len(all_data)))
    logger.info("[ヒット数]" + str(hit_count_all))
    logger.info("[ヒット率]" + str(hit_count_all / len(all_data)))

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")

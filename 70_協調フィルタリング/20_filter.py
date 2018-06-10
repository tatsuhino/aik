#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    モデルの構築
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
# 協調フィルタリング用ライブラリ
from surprise import Reader, Dataset
from surprise import SVD
# 交差検証用
import pandas
import pickle

# 定数
BASE_DIR = "./70_協調フィルタリング"

# グローバル変数
logger = getLogger(__name__)
model = pickle.load(open(BASE_DIR + "/model", 'rb')) # 学習済モデルの読み込み

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

# 指定ユーザへのおすすめアイテムの上位N件を取得
def get_predict_item_top_n(user_id,item_list, n):
    predict_item_dic = {}
    for item_id in item_list:
        item_id_formatted = '{:07d}'.format(item_id)
        predict_item_dic[item_id_formatted] = model.predict(uid=user_id, iid=item_id_formatted).est
    
    return sorted(predict_item_dic.items(), key=lambda x:x[1], reverse=True)[:n]

def evaluate(dataset,top_n):
    # 全てのデータを使って学習
    trainset = dataset.build_full_trainset()
    
    # ユーザの各アイテムの評価値を予測する
    for user_id in trainset.all_users():
        user_id_formatted = '{:07d}'.format(user_id)
        print("user:",user_id_formatted," " ,get_predict_item_top_n(user_id_formatted,trainset.all_items(),top_n))

def main():
    data_file = BASE_DIR + '/events.csv_converted'
    reader = Reader(line_format='user item rating', sep=' ')
    dataset = Dataset.load_from_file(data_file, reader=reader)
    evaluate(dataset,10)

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
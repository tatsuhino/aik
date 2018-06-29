#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    モデル学習用のデータ整形処理
    評価回数が少ないものを除去して_convertedを出力するためのおためし版。
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
import collections
from collections import defaultdict

# 定数
BASE_DIR = "./70_協調フィルタリング"

# グローバル変数
logger = getLogger(__name__)

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

# ユーザのイベント(view,addtocart)をスコア値に変換する
def eval_score(all_data):
    # viewが1回：1
    # viewが2回～3回：2
    # viewが4回以上：3
    # addtocartが1回：4
    # addtocartが2回以上：5

    for val in all_data.values():
        view_count = val["event"].count("view")
        buy_count = val["event"].count("addtocart")

        if buy_count >= 2:
            val["score"] = 5
        elif buy_count == 1:
            val["score"] = 4
        elif view_count >= 3:
            val["score"] = 3
        elif view_count >= 2:
            val["score"] = 2
        else:
            val["score"] = 1

    return all_data.values()

# Datasetロード用に'ユーザID アイテムID 評価値'のフォーマットへ変換してファイルに出力する
def convert(input_file_name):
    all_data = {}
    with open(input_file_name, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            columns = line.strip().split(',')
            if columns[1] == 'transaction' : continue 
            key = columns[0] + "_" + columns[2]

            all_data.setdefault(key, {})
            all_data[key]["user_id"] = columns[0]
            all_data[key]["item_id"] = columns[2]
            all_data[key].setdefault("event", [])
            all_data[key]["event"].append(columns[1])

    scored_data = eval_score(all_data)

    # 評価回数が１０回未満のユーザとアイテムを除外リストとして用意する
    user_event_list=[]
    item_event_list=[]
    for line in scored_data : 
        user_event_list.append(line["user_id"])
        item_event_list.append(line["item_id"])
    user_count = collections.Counter(user_event_list)
    except_user=[user for user,count in user_count.items() if count <= 10]
    item_count = collections.Counter(item_event_list)
    except_item=[item for item,count in item_count.items() if count <= 10]

    # ファイル出力
    output_file_name = input_file_name + '_converted' # 変換後のファイル名
    with open(output_file_name, mode='w') as f:
        for line in eval_score(all_data) : 
            if line["user_id"] in except_user: continue
            if line["item_id"] in except_item: continue
            f.write('{0:07d} {1:07d} {2:01d}\n'.format(int(line["user_id"]), int(line["item_id"]), int(line["score"])))

def main():
    convert(BASE_DIR + './events.csv')

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
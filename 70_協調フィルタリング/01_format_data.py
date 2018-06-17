#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    データの整形
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
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
        elif view_count >= 4:
            val["score"] = 3
        elif view_count >= 2:
            val["score"] = 2
        else:
            val["score"] = 1

    return all_data.values()

# Datasetロード用に'ユーザID アイテムID 評価値'のフォーマットへ変換してファイルに出力する
def convert(input_file_name):
    output_file_name = input_file_name + '_converted' # 変換後のファイル名

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

    # ファイル出力
    with open(output_file_name, mode='w') as f:
        for line in eval_score(all_data) : 
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
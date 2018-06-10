#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    transa
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger

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
def conv_to_score(event_str):
    return '1' if event_str  == 'view' else '2'

# Datasetロード用に'ユーザID アイテムID 評価値'のフォーマットへ変換してファイルに出力する
def convert(input_file_name):
    output_file_name = input_file_name + '_converted' # 変換後のファイル名
    output_list = []

    with open(input_file_name, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            columns = line.strip().split(',')
            if columns[1] == 'transaction' : continue 

            user_id = columns[0]
            item_id = columns[2]
            event = conv_to_score(columns[1])
            # TODO viewとadd_to_cartが両方あった場合の考慮
            output_list.append('{0:07d} {1:07d} {2:01d}\n'.format(int(user_id), int(item_id), int(event)))
              
    output_list_uniq = list(set(output_list))
    output_list_uniq.sort()
    
    # ファイル出力
    with open(output_file_name, mode='w') as f:
        for line in output_list_uniq : f.write(line)

def main():
    convert(BASE_DIR + './events.csv')

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
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
BASE_DIR = "./5_Doc2Vec_サンプル"

# グローバル変数
logger = getLogger(__name__)

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

# 
def convert(input_file_name):
    output_file_name = input_file_name + '_converted' # 変換後のファイル名

    all_data = []
    with open(input_file_name, mode='r') as f:
        lines = f.readlines()
        action_history = {}
        preUserId = ""

        for line in lines:
            columns = line.strip().split(',')
            if columns[1] == 'transaction' : continue
            if columns[0] == "" : preUserId = columns[0] # 最初の１回
            if columns[0] != preUserId: action_history = {}

            action_history["user_id"] = '{0:07d}'.format(int(columns[0]))
            action_history.setdefault("action_to_buy", [])
            action_history["action_to_buy"].append('{0:07d}'.format(int(columns[2])))
            
            if  len(set(action_history["action_to_buy"])) > 2 and columns[1] == "addtocart":
                all_data.append(action_history)
                action_history = {}
            elif len(action_history["action_to_buy"]) == 2 and columns[1] == "addtocart":
                action_history = {}
            
            preUserId = columns[0]

    all_data = sorted(all_data, key=lambda x:x["user_id"])
    # ファイル出力
    with open(output_file_name, mode='w') as f:
        i = 0
        for history in all_data : 
            f.write('{},{},{}\n'.format(i,history["user_id"]," ".join(history["action_to_buy"])))
            i += 1

def main():
    convert(BASE_DIR + './events.csv')

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
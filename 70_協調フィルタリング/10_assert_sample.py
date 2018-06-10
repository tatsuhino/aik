#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    モデルの構築から検証までを一括して行う。
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
# 協調フィルタリング用ライブラリ
from surprise import Reader, Dataset
from surprise import SVD

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

# 指定ユーザへのおすすめアイテムの上位N件を取得
def get_predict_item_top_n(model,user_id,item_list, n):
    predict_item_dic = {}
    for item_id in item_list:
        item_id_formatted = '{:07d}'.format(item_id)
        predict_item_dic[item_id_formatted] = model.predict(uid=user_id, iid=item_id_formatted).est
    
    # item_idとスコアのリスト [('0000002', 1.4807729911119272), ('0000003', 1.4807729911119272)]
    sorted_dic = sorted(predict_item_dic.items(), key=lambda x:x[1], reverse=True)[:n]
    # item_idのみのリストとして返却
    return [item_id_tapple[0] for item_id_tapple in sorted_dic]

def main():
    data_file = BASE_DIR + './events.sample.csv_converted'
    reader = Reader(line_format='user item rating', sep=' ')
    dataset = Dataset.load_from_file(data_file, reader=reader)
    all_item_set = dataset.build_full_trainset()

    # 交差検証
    dataset.split(n_folds=10,shuffle=True)
    for trainset, testset in dataset.folds():
        logger.info("[交差検証　START]------------------------------------------------")
        # modelの作成 TODO:分類機はいろいろ精度を試してみる
        model = SVD()
        model.fit(trainset)

        # TODO １ユーザに対して最大でもX回までしか評価しないようにtestセットから除く
        hit_count=0
        for test_data in testset:
            user_id = '{:07d}'.format(int(test_data[0]))
            item_id = '{:07d}'.format(int(test_data[1]))
            score = test_data[2]

            if score == 1 : continue # viewならスキップ
            # add_to_cartしている場合、そのアイテムがユーザのおすすめアイテムに入っていればHITとカウントする。
            predict_item = get_predict_item_top_n(model,user_id,all_item_set.all_items(),10)
            # print("predict_item:",predict_item, " item_id:",item_id)

            if item_id in predict_item: hit_count += 1
     
        print("評価履歴数:",str(len(testset)))
        print("ヒット数:",str(hit_count))
        print("ヒット率:",hit_count / len(testset))
        
if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
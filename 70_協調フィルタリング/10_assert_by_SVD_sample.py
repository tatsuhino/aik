#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    scikit-surpriseのドキュメントを参考に実装
    model.testメソッドで評価行列を得ている？大量のメモリが必要。
"""

# 共通
import time
from collections import defaultdict
from logging import StreamHandler, Formatter, INFO,getLogger
# 協調フィルタリング用ライブラリ
from surprise import Reader, Dataset
from surprise import SVD
from concurrent.futures import ThreadPoolExecutor

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

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def is_hit(predictions, user_id,item_id):
    predict_start = time.time()
    top_n = get_top_n(predictions, n=10)
    logger.debug("[PREDICT TIME]:{0:.5f}".format(time.time() - predict_start) + "(sec)")
    logger.info("user_id:"+user_id+" buy_item_id:"+item_id+" predict_items:" + str(top_n[user_id]))
    if item_id in top_n[user_id]: return True
                
def main():
    data_file = BASE_DIR + './events.sample.csv_converted'
    data_file = BASE_DIR + './events.csv_converted'
    reader = Reader(line_format='user item rating', sep=' ')
    dataset = Dataset.load_from_file(data_file, reader=reader)
    all_item_set = dataset.build_full_trainset()

    # 交差検証
    dataset.split(n_folds=10,shuffle=True)
    for trainset, testset in dataset.folds():
        logger.info("[交差検証　START]------------------------------------------------")
        # modelの作成 TODO:分類機はいろいろ精度を試してみる
        train_start = time.time()
        model = SVD()
        model.fit(trainset)
        logger.info("[TRAIN TIME]:{0:.5f}".format(time.time() - train_start) + "(sec)")
        logger.info("モデル学習完了")

        # 予測セットの印字
        predictions = model.test(trainset.build_anti_testset())
        # for uid, user_ratings in top_n.items():
        #     print(uid, [iid for (iid, _) in user_ratings])
        hit_count=0
        futures = []
        # TODO 結局マルチスレッドでも処理速度は上がらなかったため除去
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="thread") as executor:
            for test_data in testset:
                user_id = '{:07d}'.format(int(test_data[0]))
                item_id = '{:07d}'.format(int(test_data[1]))
                score = test_data[2]
                if score <= 3 : continue # add_to_cart以外ならスキップ
                # add_to_cartしている場合、そのアイテムがユーザのおすすめアイテムに入っていればHITとカウントする。
                futures.append(executor.submit(is_hit,predictions,user_id,item_id))

            # マルチスレッド処理の待ち合わせ
            for f in futures:
                if f.result() : hit_count += 1

        # 評価の印字
        test_buy_item_count = len([x for x in testset if x[2]==2])
        logger.info("[評価履歴数]" + str(test_buy_item_count))
        logger.info("[ヒット数]" + str(hit_count))
        if test_buy_item_count!=0 : 
            logger.info("[ヒット率]" + str(hit_count / test_buy_item_count))
        
if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")

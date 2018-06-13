#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    モデルの構築から検証までを一括して行う。
行列因子分解（Matrix Factorization）モデル
    ユーザ・アイテム行列をユーザ行列（user × k）とアイテム行列（item × k）に分解する際、既に値があるセルの値の誤差が最小になるようにする
        分解した後の行列の積をとって元に戻した際、値の入っていなかったセルに値が入っており、その値を評価値とする
    特異値分解（SVD：Singular Value Decomposition）
    非負値行列因子分解（NMF：Non-negative Matrix Factorization）
        SVDと異なり、分解した行列の要素が全て正の数
        交互最小二乗法（ALS：Alternative Least Squares）や確率的勾配降下法（SGD：Stochastic Gradient Descent）を用いて実施
クラスタモデル
    嗜好が類似した利用者のグループごとに推薦をする
関数モデル
    利用者の嗜好パターンから，アイテムの評価値を予測する関数
確率モデル
    行動分布型：どの利用者が，どのアイテムを，どう評価したかの分布をモデル化
    評価分布型：全アイテムに対する評価値の同時分布をモデル化
    ナイーブベイズ、ベイジアンネットワークなど
時系列モデル
    マルコフ過程：アイテムを評価した時間的順序も考慮
    マルコフ決定過程(MDP：Markov Decision Process)：加えて，利用者の行動もモデル化
"""

# 共通
import time
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

# 指定ユーザへのおすすめアイテムの上位N件を取得
def get_predict_item_top_n(model,user_id,item_list, n):
    predict_item_dic = {}
    for item_id in item_list:
        item_id_formatted = '{:07d}'.format(item_id)
        predict_item_dic[item_id_formatted] = model.predict(uid=user_id, iid=item_id_formatted).est
    
    # item_idとスコアのリスト [('0000002', 1.4807729911119272), ('0000003', 1.4807729911119272)]
    sorted_dic = sorted(predict_item_dic.items(), key=lambda x:x[1], reverse=True)[:n]
    logger.info("予想アイテム:"+str(sorted_dic))
    # item_idのみのリストとして返却
    return [item_id_tapple[0] for item_id_tapple in sorted_dic]

def is_hit(model, user_id,item_id,item_list):
    predict_item = get_predict_item_top_n(model,user_id,item_list,10)
    logger.info("user_id:"+user_id+" buy_item_id:"+item_id+" predict_items:" + str(predict_item))
    if item_id in predict_item: return True
                
def main():
    data_file = BASE_DIR + './events.csv_converted'
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
        logger.info("モデル学習完了")

        # TODO １ユーザに対して最大でもX回までしか評価しないようにtestセットから除く
        hit_count=0
        futures = []

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="thread") as executor:
            for test_data in testset:
                user_id = '{:07d}'.format(int(test_data[0]))
                item_id = '{:07d}'.format(int(test_data[1]))
                score = test_data[2]

                if score != 2 : continue # add_to_cart以外ならスキップ
                # add_to_cartしている場合、そのアイテムがユーザのおすすめアイテムに入っていればHITとカウントする。
                futures.append(executor.submit(is_hit,model,user_id,item_id,all_item_set.all_items()))

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
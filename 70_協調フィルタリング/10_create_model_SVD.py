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
import pickle

# 定数
BASE_DIR = "./70_協調フィルタリング"

# グローバル変数
logger = getLogger(__name__)
model = SVD() #予測アルゴリズム:SVD
'''
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
'''
# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

def save_model(dataset):
    # 全てのデータを使って学習
    trainset = dataset.build_full_trainset()
    model.fit(trainset)
    pickle.dump(model, open("model", 'wb'))

def main():
    data_file = BASE_DIR + './events.csv_converted'
    reader = Reader(line_format='user item rating', sep=' ')
    dataset = Dataset.load_from_file(data_file, reader=reader)
    save_model(dataset)

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    寿司のレコメンド scikit-surprise使用版
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
# 協調フィルタリング用ライブラリ
from surprise import Reader, Dataset
from surprise import SVD

# 定数
BASE_DIR = "./61_協調フィルタリング_寿司"

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

# Datasetロード用に'ユーザID アイテムID 評価値'のフォーマットへ変換してファイルに出力する
def convert(input_file_name):
    output_file_name = input_file_name + '_converted' # 変換後のファイル名
    output = ''
    
    with open(input_file_name, mode='r') as f:
        lines = f.readlines()
        user_id = 0
        for line in lines:
            scores_str = line.strip().split(' ')
            for item_id, score_str in enumerate(scores_str):
                score = int(score_str)
                if score != -1: output += '{0:04d} {1:02d} {2:01d}\n'.format(user_id, item_id, score)
                    
            user_id += 1
            
    # ファイル出力
    with open(output_file_name, mode='w') as f: f.write(output)
    return output_file_name

def evaluate(dataset):

    # 全てのデータを使って学習
    trainset = dataset.build_full_trainset()
    model.fit(trainset)

    # ユーザ間の類似度を計算
    similarities = model.compute_similarities()

    logger.info('similarities.shape: {}'.format(similarities.shape))
    # similarities.shape: (5000, 5000)
    logger.info('Similarity(User 0000 and 0001: {:.3f})'.format(similarities[0,1]))
    # Similarity(User 0000 and 0001: 0.500)

    # 0番目のユーザの0番目のアイテム(エビ)の評価値を予測する
    user_id = '{:04d}'.format(0)
    item_id = '{:02d}'.format(0)

    prediction = model.predict(uid=user_id, iid=item_id)

    logger.info('Predicted rating(User: {0}, Item: {1}): {2:.2f}'
            .format(prediction.uid, prediction.iid, prediction.est))
            
def main():
    # output_file_name = convert(BASE_DIR + './sushi3b.5000.10.score')

    data_file = BASE_DIR + './sushi3b.5000.10.score_converted'
    reader = Reader(line_format='user item rating', sep=' ')
    dataset = Dataset.load_from_file(data_file, reader=reader)
    evaluate(dataset)

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
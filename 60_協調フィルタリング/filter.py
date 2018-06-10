#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    templete.py

    プログラム全体の説明を書きます。
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
#####
import numpy as np


# 定数です
NUM_SPHERE = 4
NUM_TRYSAIL = 3
logger = getLogger(__name__)

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)


def get_correlation_coefficents(scores, target_user_index):
    similarities = []
    target = scores[target_user_index]
    
    for i, score in enumerate(scores):
        # 共通の評価が少ない場合は除外
        indices = np.where(((target + 1) * (score + 1)) != 0)[0]
        if len(indices) < 3 or i == target_user_index:
            continue
        
        similarity = np.corrcoef(target[indices], score[indices])[0, 1]
        if np.isnan(similarity):
            continue
    
        similarities.append((i, similarity))
    
    return sorted(similarities, key=lambda s: s[1], reverse=True)

def main():
    
    logger.info("mainの開始")
    scores = np.loadtxt('sushi3b.5000.10.score', delimiter=' ')
    target_user_index = 0 # 0番目のユーザ
    similarities = get_correlation_coefficents(scores, target_user_index)

    print('Similarities: {}'.format(similarities))
    # Similarities: [(186, 1.0), (269, 1.0), (381, 1.0), ...

    print('scores[186]:\n{}'.format(scores[186]))



if __name__ == "__main__":
    init_logger()
    start = time.time()
    main()
    logger.info("[TIME ALL]:{0}".format(time.time() - start) + "(sec)")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
BASE_DIR = "./80_協調フィルタリング_アイテム"

# グローバル変数
logger = getLogger(__name__)

# ログの設定
def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(INFO)

def main():
    # convert(BASE_DIR + './events.csv')
    kf = KFold(n_splits=10, shuffle=True, random_state=1 )
    DF = pd.DataFrame([{"1":"a"},{"2":"a"},{"3":"a"},{"1":"a"},{"2":"a"},{"3":"a"},{"1":"a"},{"2":"a"},{"3":"a"},{"1":"a"},{"2":"a"},{"3":"a"}])
    for train_index,test_index in kf.split(DF):
        print(test_index)

if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")



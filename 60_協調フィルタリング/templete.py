#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    templete.py

    プログラム全体の説明を書きます。
"""

import time
from logging import StreamHandler, Formatter, INFO,getLogger

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

def main():
    logger.info("mainの開始")
    func_fugafuga(1)


def func_fugafuga(parm1):
    """ なんか関数ですよ

        @param parm1: この引数はとくにつかいません
    """
    logger.error("エラーです！！")

class MyClass:

    def __init__(self):
        """コンストラクタの説明
        """
        self._pv_v = "インスタンス変数"

    def process(self, parm1):
        """ メソッドの説明
        """
        logger.debug("process")

    def _pv_process(self):
        logger.debug("_pv_process")


if __name__ == "__main__":
    init_logger()
    start = time.time()
    main()
    logger.info("[TIME ALL]:{0}".format(time.time() - start) + "(sec)")
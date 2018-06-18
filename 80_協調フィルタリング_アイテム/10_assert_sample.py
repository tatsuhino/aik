#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    モデルの構築から検証までを一括して行う。
"""

# 共通
import time
from logging import StreamHandler, Formatter, INFO,getLogger
import os
# 協調フィルタリング用ライブラリ
from surprise import Reader, Dataset
from surprise import SVD, KNNBasic, KNNBaseline
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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

# 指定ユーザへのおすすめアイテムの上位N件を取得
# def get_predict_item_top_n(model,user_id,item_list, n):
#     predict_item_dic = {}
#     for item_id in item_list:
#         item_id_formatted = '{:07d}'.format(item_id)
#         predict_item_dic[item_id_formatted] = model.predict(uid=user_id, iid=item_id_formatted).est
    
#     # item_idとスコアのリスト [('0000002', 1.4807729911119272), ('0000003', 1.4807729911119272)]
#     sorted_dic = sorted(predict_item_dic.items(), key=lambda x:x[1], reverse=True)[:n]
#     # logger.info("予想アイテム:"+str(sorted_dic))
#     # item_idのみのリストとして返却
#     return [item_id_tapple[0] for item_id_tapple in sorted_dic]
# ユーザのイベント(view,addtocart)をスコア値に変換する
def eval_score(all_data):
    for val in all_data.values():
        view_count = val["event"].count("view")
        buy_count = val["event"].count("addtocart")

        if buy_count >= 2:
            val["score"] = 5
        elif buy_count == 1:
            val["score"] = 4
        elif view_count >= 4:
            val["score"] = 3
        elif view_count >= 2:
            val["score"] = 2
        else:
            val["score"] = 1

    return all_data.values()

def is_hit(model, user_id,item_id,item_list):
    predict_item = get_predict_item_top_n(model,user_id,item_list,10)
    logger.info("user_id:"+user_id+" buy_item_id:"+item_id+" predict_items:" + str(predict_item))
    if item_id in predict_item: return True

# おすすめアイテムの上位N件                             を取得(購入直前のアイテム２つに対して、類似のアイテムを取得)
def get_predict_item_top_n(model,test_data_line, n):
    buy_item = test_data_line.pop()
    # 購入の直前に見たアイテムから操作する
    test_data_line.reverse()

    predict_dict = {}
    predict_dict["exceed_sim_vecs"] = []
    predict_dict["sim_vecs"] = []
    for item in test_data_line :
        if item == buy_item: continue
        try:
            inner_id = model.trainset.to_inner_iid(item)
            predict_item = model.get_neighbors(inner_id, k=int(n))
            predict_item = [model.trainset.to_raw_iid(inner_id)
                       for inner_id in predict_item]
            print(predict_item)

            predict_dict["exceed_sim_vecs"].extend(predict_item)
            predict_dict["sim_vecs"].extend(predict_item[:int(n/2 + n%2)]) # TODO 直近２件決め打ちなロジック
            if len(predict_dict["exceed_sim_vecs"]) >= (n*2): break
        except KeyError: continue # ボキャブラリーに該当itemが存在しない場合
        except ValueError: continue # ボキャブラリーに該当itemが存在しない場合
    
    # 予想アイテム数がnを超える前にループを抜けた場合は、exceed_sim_vecsから持ってくる
    if len(predict_dict["sim_vecs"]) < n:
        predict_dict["sim_vecs"].extend(predict_dict["exceed_sim_vecs"])
    if len(predict_dict["sim_vecs"]) == 0 : return [] 

    all_predict_item = [vec[0] for vec in predict_dict["sim_vecs"]]
    return list(set(all_predict_item))[:n]

def export_format_data(all_data_before):
    all_data = {}
    for line in all_data_before:
        for v in line.values() : line_value = v
        user_id = line_value.split(",")[0]
        item_ids = line_value.split(",")[1].split(" ")
        # view,view,addtocart
        for i, item_id in enumerate(item_ids): 
            key = user_id + "_" + item_id
            all_data.setdefault(key, {})
            all_data[key]["user_id"] = user_id
            all_data[key]["item_id"] = item_id
            all_data[key].setdefault("event", [])
            if (i+1) == len(item_ids): 
                all_data[key]["event"].append("addtocart")
            else: 
                all_data[key]["event"].append("view")

    train_tmp_file = BASE_DIR +"/tmp"
   # os.remove(train_tmp_file)
    # ファイル出力
    with open(train_tmp_file, mode='w') as f:
        for line in eval_score(all_data) : 
            f.write('{0:07d} {1:07d} {2:01d}\n'.format(int(line["user_id"]), int(line["item_id"]), int(line["score"])))
    
    return train_tmp_file

def main():
    all_data = []
    with open(BASE_DIR + './events.csv_converted', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            data_dict={}
            colums = line.split(",")
            data_dict[str(colums[0])] = colums[1] + "," + colums[2].replace('\n','')
            all_data.append(data_dict)

    rule = KFold(n_splits=10, shuffle=True, random_state=1)
    all_data_frame = pd.DataFrame(all_data)
    for train_index,test_index in rule.split(all_data_frame):
        train_data = [all_data[i] for i in train_index]
        test_data_list = [all_data[i] for i in test_index]
        train_tmp_file =  export_format_data(train_data)

        reader = Reader(line_format='user item rating', sep=' ')
        dataset = Dataset.load_from_file(train_tmp_file, reader=reader)
        trainset = dataset.build_full_trainset()

        # sim_options = { 'name': 'cosine',
        #                 'user_based': False }
        # model = KNNBasic(sim_options=sim_options)
        # model.fit(trainset)

        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        model = KNNBaseline(sim_options=sim_options)
        model.fit(trainset)
        # pickle.dump(model, open("model", 'wb'))
        # loaded_model = pickle.load(open("model, 'rb'))
        similarities = model.compute_similarities()
        logger.info("モデル学習完了")

        hit_count=0
        for test_data in test_data_list:
            for v in test_data.values() : line_value = v
            test_line = line_value.split(",")[1].split(" ")
            buy_item_id = test_line[-1]
            # pre_buy_item_id = line_value.split(",")[1].split(" ")[-2]

            predict_list = get_predict_item_top_n(model, test_line,10)
            if buy_item_id in predict_list : hit_count += 1

        # 評価の印字
        test_buy_item_count = len(test_data_list)
        logger.info("[評価履歴数]" + str(test_buy_item_count))
        logger.info("[ヒット数]" + str(hit_count))
        if test_buy_item_count!=0 : 
            logger.info("[ヒット率]" + str(hit_count / test_buy_item_count))

        return
        
if __name__ == "__main__":
    init_logger()
    start = time.time()
    logger.info("[START]------------------------------------------------")
    main()
    logger.info("[TIME]:{0:.2f}".format(time.time() - start) + "(sec)")
    logger.info("[END]--------------------------------------------------")
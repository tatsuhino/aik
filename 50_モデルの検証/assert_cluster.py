#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python assert_cluster.py dev_1/history.1.txt dev_1/model cluster/svc.pkl.cmp_1000 cluster/cluster_result.csv 

import csv
import collections
import argparse
import gensim
from gensim import models
from gensim.models.doc2vec import TaggedDocument

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('model')
parser.add_argument('kmean_model')
parser.add_argument('kmean_mapping')
parser.add_argument('--predict_count','-p', default=5, type=int)
args = parser.parse_args()

model = models.Doc2Vec.load(args.model)
kmeans_model = models.Doc2Vec.load(args.kmean_model)

#クラスタIDと予想購入アイテムのmapを取得する
def get_cluster_predict_dict(predict_count):
    cluster_predict_dict = {}
    with open(args.kmean_mapping, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        cluster_id = row[0]
        doc_ids = row[1]
        # クラスタと予想アイテムとのマッピング
        predict_item_all = [doc_id.split("_")[0] for doc_id in doc_ids.split(" ")]
        predict_item_top = collections.Counter(predict_items).most_common(predict_count)
        cluster_predict_dict[cluster_id] = predict_item_top5
    
    return cluster_predict_dict
  
 #クラスタIDと予想購入アイテムのmapを取得する
def get_cluster_id(history_vector):
    return kmeans_model.predict(history_vector)
   

#履歴データ一行分のデータを格納するためのクラス
class HistoryRow ():

  def __init__(self, history_line):
    self.history_items = history_line.replace('\n','').split(" ")
        
  # 閲覧絵ベクトルを取得
  def get_history_vector(self,model):
    return model.infer_vector(self.__get_view_items())
 
  # 購入アイテムを取得
  def get_buy_item(self):
    return self.history_items[-1]
      
  # 閲覧アイテムを取得
  def __get_view_items(self):
    return self.history_items[0:-1]

#評価値を算出する
def print_model_accuracy(history_table,predict_items):
  target_history_table = []
    
  hit_count=0
  for row in target_history_table:
  #  print("buy:",row.get_buy_item() ,"pre:",row.get_predict_items(model))
    if row.get_buy_item() in predict_items:
      hit_count = hit_count + 1
      
  print("評価履歴数:",str(len(target_history_table)))
  print("ヒット数:",str(hit_count))
  print("ヒット率:",hit_count / len(target_history_table))

history_table = []
predict_dict = get_cluster_predict_dict(args.predict_count)

#history.X.txtの検証データを受け取り、assert_resultに類似ベクトルデータを書き込み
file = open('assert_result_memo.csv','w')
with open(args.input, 'r') as f:
  for line in f:
    row = HistoryRow(line,args.predict_count)
    history_table.append(row)
    
    row_cluster_id = get_cluster_id(row.get_history_vector(model))
    predict_items = predict_dict[row_cluster_id]

    #印字用
    file.write(' '.join(predict_items))
    file.write(",")
    file.write(str(row.get_buy_item()))
    file.write(",")
    file.write(line)

print_model_accuracy(history_table,predict_items)
file.close()

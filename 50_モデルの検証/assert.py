#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gensim
from gensim import models
from gensim.models.doc2vec import TaggedDocument

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='D:\model_1\history.1.txt')
parser.add_argument('--model', default='D:\model_1\model')
parser.add_argument('--predict_count','-p', default=5, type=int)
parser.add_argument('--history_min_count','-m', default=1, type=int)
args = parser.parse_args()

model = models.Doc2Vec.load(args.model)
f = open('noise_doc_tag.txt')
noise_list = f.read().split(",")  # ファイル終端まで全て読んだデータを返す
f.close()


#履歴データ一行分のデータを格納するためのクラス
class HistoryRow ():

  def __init__(self, history_line, predict_count):
    self.history_items = history_line.replace('\n','').split(" ")
    self.predict_count = predict_count
        
  # 閲覧ベクトルの重複を除いた数
  def get_view_item_count(self):
    return len(set(self.history_items))
    
  # モデルをもとに予測される購入アイテムリストを返却する。
  def get_predict_items(self,model):
    history_vec = model.infer_vector(self.__get_view_items())
    sim_vecs = model.docvecs.most_similar([history_vec],topn=self.predict_count)
   
    list = []
    for vec in sim_vecs:
      list.append(vec[0].split("_")[0])
    return list
      
  # 購入アイテムを取得
  def get_buy_item(self):
    return self.history_items[-1]
      
  # 閲覧アイテムを取得
  def __get_view_items(self):
    return self.history_items[0:-1]

#評価値を算出する
def print_model_accuracy(history_table,history_min_count):
  target_history_table = []
  
  for row in history_table:
    if row.get_view_item_count() >= history_min_count:
      target_history_table.append(row)
    
  hit_count=0
  for row in target_history_table:
    predict_item = row.get_predict_items(model)
    # predict_item_noise_removed = [x for x in predict_item if x not in noise_list]
    if row.get_buy_item() in predict_item:
      hit_count = hit_count + 1
     
  print("評価履歴数:",str(len(target_history_table)))
  print("ヒット数:",str(hit_count))
  print("ヒット率:",hit_count / len(target_history_table))

history_table = []
#assert.txtの検証データを受け取り、assert_resultに類似ベクトルデータを書き込み
file = open('assert_result.csv','w')
with open(args.input, 'r') as f:
  for line in f:
    row = HistoryRow(line,args.predict_count)
    history_table.append(row)
    #印字用
    file.write(' '.join(row.get_predict_items(model)))
    file.write(",")
    file.write(str(row.get_view_item_count()))
    file.write(",")
    file.write(str(row.get_buy_item()))
    file.write(",")
    file.write(line)

print_model_accuracy(history_table,args.history_min_count)
file.close()
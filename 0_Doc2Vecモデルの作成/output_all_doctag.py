#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim 
from gensim import models
from collections import defaultdict

model = models.Doc2Vec.load('D:\model\model')


def get_target_doc_list(min_doc_count):
  all_doc_list = []
  for row in model.docvecs.doctags:
    all_doc_list.append(row)

  # 全docタグのmapを作成
  all_doc_dict = defaultdict(lambda: list())
  for doc_id in all_doc_list:
    key = doc_id.split("_")[0]
    all_doc_dict[key].append(doc_id)

  # 要素数min_doc_count以上のdoc_idのリストを作成
  target_doc_list = []
  for doc_list in all_doc_dict.values():
    if (len(doc_list) >= min_doc_count):
      target_doc_list.append(doc_list)
  return target_doc_list







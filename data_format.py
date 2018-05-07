#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

def createCsv(file,name, body):
  line = ' '.join(body)
  print(line)
  file.write(line)
  file.write('\n')


file = open('output_blank.txt','a')
with open('log_cate.csv', 'r') as f:
  reader = csv.reader(f)
  header = next(reader)
  
  viewEventList = []
  preUserId = ""
  
  # 1行ごとの処理
  for row in reader:
    userId = row[0]
    eventId = row[1]
    actionType = row[2]
    
    # ユーザIDが変わってたらリストをクリア
    # print('preUserId:',preUserId,'userId:',userId,'actionType:',actionType,'viewEventList:',viewEventList)
    if userId !=preUserId:
      viewEventList = []

    viewEventList.append(eventId)
    
    # 購入したらリストを印字
    if userId != "" and len(viewEventList) > 1 and actionType=="3":
      createCsv(file,userId,viewEventList)
      viewEventList = []
    elif len(viewEventList) == 1 and actionType=="3":
      viewEventList = []
    
    preUserId = userId
    
file.close()
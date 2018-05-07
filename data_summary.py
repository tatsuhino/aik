#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv

def initSummaryDict(dict):
  with open('mapping.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    for row in reader:
      eventId = row[0]
      cateId = row[1]
      dict[eventId] = cateId

outputfile = open('log_cate.csv','a')
event_cate_dict = {}
initSummaryDict(event_cate_dict)
print(event_cate_dict)

with open('log.csv', 'r') as f:
  reader = csv.reader(f)
  header = next(reader)

  # 1çsÇ≤Ç∆ÇÃèàóù
  for row in reader:
    userId = row[0]
    eventId = row[1]
    actionType = row[3]

    try: 
      cateId = event_cate_dict[eventId]
    except KeyError:
      print("skip:",eventId)
      continue

    body = [userId, cateId, actionType]
    line = ','.join(body)
    outputfile.write(line)
    outputfile.write('\n')

outputfile.close()
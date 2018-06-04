#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file = open('out.txt','w')

top_item=[]
with open('D:\model_1\history.1.txt', 'r') as f:
  for line in f:
    top_item.append(line.replace('\n','').split(" ")[-1])

print(top_item)
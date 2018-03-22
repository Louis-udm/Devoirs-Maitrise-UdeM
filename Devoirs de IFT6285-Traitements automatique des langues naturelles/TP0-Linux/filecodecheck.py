# # # #
# decoding.py
# @author Zhibin.LU
# @created Mon Jan 15 2018 21:13:58 GMT-0500 (EST)
# @last-modified Thu Jan 18 2018 10:15:21 GMT-0500 (EST)
# @website: https://louis-udm.github.io
# @description 
# # # #

import chardet
import sys
import os
print sys.version
# print sys.version_info #代码中对比用
# print sys.path
print os.getcwd()
with open('./TP/zola1.txt','r') as f:
    result = chardet.detect(f.read())
    print result

with open('./TP/zola1.cand.txt','r') as f:
    result = chardet.detect(f.read())
    print result

str='À la mère. à la mÈRE. Êtes-vous français? êtres-vous franÇAIS?'
print str
print(str.upper())          # 把所有字符中的小写字母转换成大写字母
print(str.lower())          # 把所有字符中的大写字母转换成小写字母
print(str.capitalize())     # 把第一个字母转化为大写字母，其余小写
print(str.title())          # 把每个单词的第一个字母转化为大写，其余小写

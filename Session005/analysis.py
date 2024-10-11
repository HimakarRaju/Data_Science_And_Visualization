# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:28:26 2024

@author: HimakarRaju
"""

s = 'TextData.txt'
n = input("text to remove : ")

with open(s, "r") as file:
    lines = file.readlines()
    n = input('Enter the string:')
    newlines = []
    for line in lines:
        if line != n:
            newlines.append(line)
        else:
            continue
with open(s, "w") as file:
    file.writelines(newlines)

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:18:04 2020

@author: ashba
"""
try:
    from env.possible_moves import possible_moves
except:
    from possible_moves import possible_moves
def creat_white_promotion_moves():
    
    one_a= []
    b= possible_moves
    for i in b:
        i = i[:1]+'2' + i[2:3] + '1' +i[4:]
        #i = i[:3]+'1'+ i[4:]
        one_a.append(i)
def all_possible_moves():
    one_a= []
    for i in possible_moves:
        if len(i) <= 4:
            continue
        else:
            #print(len(i))
            if i[4] == 'q' or i[4] == 'b' or i[4] == 'r' :
                i = i[:1]+'2' + i[2:3] + '1' +i[4:]
                #i = i[:3]+'1'+ i[4:]
                one_a.append(i)
    for t in one_a:
        possible_moves.append(t)
        
        
import random

"""
this function generate random moves from a given set of available moves
"""
def choose_random_moves(moves: list) -> str:
    return random.choice(moves)

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:43:09 2020

@author: ashba
"""



def available_move(d):
    available_moves = []
    for i in d:
        available_moves.append(i.uci())
    return available_moves


def check_valid(env, action):
    fen = list(env.board.legal_moves)
    av_mv = available_move(fen)
    if action in av_mv:
        return av_mv, True
    return av_mv, False
    

def make_move(env, action):    
    reward = 0
    gameover = True
    av, check = check_valid(env, action)
    if check:
        env.step(action)
    else:
        print("action is invalid: ", action)
        print("\n valid moves is ", av)
        a = input("plase enter valid move:  ")
        make_move(env, a)
    if env.board.result() == '*':
        reward = 0
        gameover = False
        
    else:
        if env.result == '1-0':
            reward = 1
        elif env.result == '0-1':
            reward = -1
        else:
            reward = 0.5
    
    return reward, gameover


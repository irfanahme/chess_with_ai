# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:59:59 2020

@author: ashba
"""
import tensorflow as tf
ver = tf.version.VERSION
if (int(ver[0])>1):
    print (True)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


import os, random


import env.chess_env as chess_env
from DQN.net_function_tf_k  import creat_network_3d
from DQN.DQN_player import DQN_Bot

def play_bot_to_bot(env, player1, player2):
   """white is player1
    black is player2"""
   while env.board.result() == "*":
#        print(env.board)
        if env.white_to_move:
            action = player1.choose_move(env)
        else:
            action = player2.choose_move(env)
#        print(action)

if __name__ == "__main__":
 
    training_episodes = 200000
    game_draw=White=Black=0
    cent_episode = []
    b1_path= "models/b1_ckpts_at" + str(training_episodes) + "steps/"
    b2_path ="models/b2_ckpts_at" + str(training_episodes)+ "steps/"
    if not os.path.exists(b1_path) :
        os.mkdir(b1_path)
    if not os.path.exists(b2_path) :
        os.mkdir(b2_path)
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    if tf.test.is_gpu_available():
        device= "/gpu:6"
    else:
        device = "/cpu:0"
    with tf.device(device):
       sess1 = tf.Session(config = config)
       optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
       b1=DQN_Bot( creat_network_3d, optimizer1, sess1,b1_path)
 
    if tf.test.is_gpu_available():
        device= "/gpu:7"
    else:
        device = "/cpu:0"
    with tf.device(device):
       sess2 = tf.Session(config = config)       
       optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001)
       b2 =DQN_Bot(creat_network_3d, optimizer2,sess2, b2_path)
    b1_var = b1.var
    b2_var = b2.var 
    print("\n\n\n\n\n\n",  len(b2_var),len(b1_var))
   
    c = []
    for i in range(len(b2_var)):
         if b2_var[i] not in b1_var:
             c.append(b2_var[i])
    b2.var = c
    print(len(c))
    print( len(b2_var),len(b1_var)) 
    b1.set_saver(b1.var)  
    b2.set_saver(b2.var)  
    with tf.device("/gpu:5"): 
       env = chess_env.ChessEnv()
       for i_episode in range(training_episodes):
         env.reset()
         flip = random.randint(0, 1)
         if (flip == 0):
              play_bot_to_bot(env, b1,b2)
         else:
               play_bot_to_bot(env, b2,b1)
        
    
         if env.result == '1-0':
             White +=1
         elif env.result == '0-1':
             Black +=1
         else:
             game_draw  +=1
         
         if i_episode % 100 == 0:
            print("\n\n\nEpisode {}".format(i_episode))
            print("\twhite:" + str(White))
            print("\tblack:" + str(Black))
            print("\tdraw:" + str(game_draw))
            x =(White, Black, game_draw)
            cent_episode.append(x)
            game_draw=White=Black=0
         if i_episode % 1000 == 0: 
            b1.save_ckpt(b1_path, i_episode)
            b2.save_ckpt(b2_path, i_episode)


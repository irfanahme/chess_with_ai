import tensorflow as tf
ver = tf.version.VERSION
if (int(ver[0])>1):
    print (True)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


#import tensorflow.compat.v1 as tf
import time
import env.chess_env as chess_env
from DQN.net_function_tf_k import creat_network_3d
from DQN.DQN_player import DQN_Bot
from env.move_func import make_move, available_move
#from DQN.deep_q_network import DeepQNetwork
import os , random
import tensorflow as t
#from DQN.dqn_V import DeepQNetwork
from mcts import MCTSPlayer, evaluate


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001)
i = 1
def human_choose_move(env):
    valid_move = available_move(list(env.board.legal_moves))
    print(valid_move)
    a = input("inter the move  :")
    make_move(env, str(a))
    print(env.board)

def bot_choose_move(env, action):
    #valid_move = available_move(list(env.board.legal_moves))
    #print(valid_move)
    #a = input("inter the move  :")
    make_move(env, str(action))
    print(env.board)


def play_bot_to_bot(env):
    b = DQN_Bot(creat_network_3d, optimizer1)
    b2 = DQN_Bot(creat_network_3d, optimizer2)
    
    while env.board.result(claim_draw = True) == "*":
        print(env.board)
        if env.white_to_move:
            action = b.choose_move(env)
        else:
            action = b2.choose_move(env)
        print(action)

def play_human_to_bot(env, p, b, b1):
    global i
    #sess = sess = tf.Session()
    '''q_network1 = DeepQNetwork(sess,
                             optimizer1,
                             creat_network_3d,
                             state_dim = (18,8,8) ,
                             num_actions = 1968,
                             batch_size=32,
                             init_exp=0.6,         # initial exploration prob
                             final_exp=0.01,   #0.1     # final exploration prob
                             anneal_steps=120000,  # N steps for annealing exploration
                             double_q_learning=True,
                             discount_factor=0.8)'''

    #training_episodes = 200000
    #bl_path ="models/white_ckpts_at" + str(training_episodes)+ "steps/"
    #black_path ="models/black_ckpts_at" + str(training_episodes)+ "steps/"
    #optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    #b = DQN_Bot(creat_network_3d, optimizer1,sess,black_path, block = 2)
    #b1=DQN_Bot(creat_network_3d, optimizer1, sess,bl_path,  block = 2)
    #env = chess_env.ChessEnv()

    #b2 = MCTSPlayer(evaluate)
    if p == "WHITE":
        #b2 = MCTSPlayer(evaluate)
        print("num_halfmoves = "env.num_halfmoves)
        while env.board.result(claim_draw = True) == "*":
            #global i
            #i += 1
            #print(i)
            #print(env.board)
            if env.black_to_move:
                #print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<>>>>> white")
                print("white player move")
                b2 = MCTSPlayer(evaluate, n_playout = 5)
                action, probs = b2.get_action(env, b1)
                bot_choose_move(env, action)
                i += 1
                #print('white>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i)
                print(">>>>>>>>>>>>>>>>>>env.bot_player", env.bot_player)
                print(action)
                time.sleep(5)
                
            else:
                #action = b.choose_move(env)
                human_choose_move(env)
                print(">>>>>>>>>>>>>>>>>>env.bot_player"", env.bot_player)
#                print(action)
#                print(action)

    else:
       while env.board.result(claim_draw = True) == "*":
            #global i
            #i += 1
            #print(i)a
            #print('"""""""""""""""""""""""""""""""""">>>>>>>>>>>>>>>>>>>>>>>>>>>Black')
            #print(env.board)
            if env.white_to_move:
                b2 = MCTSPlayer(evaluate, n_playout = 5)
                action, probs = b2.get_action(env, b1)
                bot_choose_move(env, action)
                i += 1
                print(">>>>>>>>>>>>>>>>>env.bot_player"",env.bot_player)
                print('black>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',i)
                time.sleep(5)
            else:
                human_choose_move(env)
                print(">>>>>>>>>>>>>>>>>>env.bot_player"", env.bot_player)
#                print(action)
                time.sleep(5)
    


def play_human_to_human(env, p):
    if p == "WHITE":
        while env.board.result() == "*":
            print(env.board)
            if env.white_to_move:
                print("white player move")
                human_choose_move(env)
                
            else:
                print("Black player move")
                human_choose_move(env)

    else:
        while env.board.result() == "*":
            print(env.board)
            if env.black_to_move:
                print("black player move")
                human_choose_move(env)

            else:
                print("white player move")
                human_choose_move(env)


sess = sess = tf.Session()
'''q_network1 = DeepQNetwork(sess,
                             optimizer1,
                             creat_network_3d,
                             state_dim = (18,8,8) ,
                             num_actions = 1968,
                             batch_size=32,
                             init_exp=0.6,         # initial exploration prob
                             final_exp=0.01,   #0.1     # final exploration prob
                             anneal_steps=120000,  # N steps for annealing exploration
                             double_q_learning=True,
                             discount_factor=0.8)'''
training_episodes = 200000
bl_path ="models/white_ckpts_at" + str(training_episodes)+ "steps/"
black_path ="models/black_ckpts_at" + str(training_episodes)+ "steps/"
#optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
b = DQN_Bot(creat_network_3d, optimizer1,sess,black_path, block = 2)
b1=DQN_Bot(creat_network_3d, optimizer1, sess,bl_path,  block = 2)


for i in range (5):
       env = chess_env.ChessEnv()
       env.reset()    
       print("Welcome to B vs H")
       p = input("Select Your Player    :")
       p = str.upper(p)
       print("player ............................., ", p)
       player = ["WHITE", "BLACK"]
       if p == player[0]:
#                env.change_turn(player[1])
          play_human_to_bot(env, p, b, b1)
       else:
#                env.change_turn(player[0])
          play_human_to_bot(env ,p, b, b1)

import env.chess_env as chess_env
from DQN.net_function_tf_k  import creat_network_3d
from DQN.DQN_player import DQN_Bot
import os , random
import tensorflow as tf
#from DQN.dqn_V import DeepQNetwork
from mcts import MCTSPlayer, evaluate
from env.move_func import make_move, available_move



def human_choose_move(env):
    valid_move = available_move(list(env.board.legal_moves))
    print(valid_move)
    a = input("inter the move  :")
    make_move(env, str(a))
#    print(env.board)




def play_bot_to_bot(env, player1, player2):
   """white is player1
    black is player2"""
   while env.board.result() == "*":
#        print(env.board)
        if env.white_to_move:
            action = player1.choose_move(env)
        else:
            action = player2.choose_move(env)
training_episodes = 200000
game_draw=White=Black=0
cent_episode = []
b1_path= "models/b1_ckpts_at" + str(training_episodes) + "steps/"
b2_path = "models/b2_ckpts_at" + str(training_episodes) + "steps/"
if not os.path.exists(b1_path) :
    os.mkdir(b1_path)
if not os.path.exists(b2_path) :
    os.mkdir(b2_path)
sess1 = tf.Session()

optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
b1=DQN_Bot( creat_network_3d, optimizer1, sess1,b1_path)
env = chess_env.ChessEnv()
#b = MCTSPlayer(evaluate)
#action, probs = b.get_action(env, b1)


while env.board.result() == "*":
        print(env.board)
        if env.white_to_move:
            b = MCTSPlayer(evaluate)
            action, probs = b.get_action(env, b1)
            reward, gameover  = make_move(env, action)

        else:
            human_choose_move(env)


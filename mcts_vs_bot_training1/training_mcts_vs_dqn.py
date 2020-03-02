import tensorflow as t
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import env.chess_env as chess_env
from DQN.net_function_tf_k import creat_network_3d
from DQN.DQN_player import DQN_Bot
from env.move_func import make_move, available_move
#from DQN.deep_q_network import DeepQNetwork
import os , random
import tensorflow as t
from DQN.dqn_V import DeepQNetwork


from DQN.deep_q_network import DeepQNetwork
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import env.chess_env as chess_env

def choose_random_moves(moves: list) -> str:
    return random.choice(moves)
def choose_move(possible_moves: list) -> str:
        return choose_random_moves(possible_moves)

def play_bot_to_bot(env, player1, player2):
   """white is player1
    black is player2"""
   while env.board.result(claim_draw=True) == "*":
#        print(env.baoard)
        if env.white_to_move:
            with tf.device("/gpu:1"):
               player1.choose_move_mcts(env, player1)
        else:
            #valid_move = list(env.board.legal_moves)
            #action = choose_move(valid_move)
            #env.step(str(action))
            with tf.device("/gpu:0"):
               action = player2.choose_move(env)
def play_dqn_to_mcts_dqn(env, player1, player2):
   """white is player1
    black is player2"""
   while env.board.result(claim_draw=True) == "*":
#        print(env.board)
        if env.white_to_move:
            with tf.device("/gpu:3"):
                 action = player1.choose_move(env)
            #player1.choose_move_mcts(env, player1)
        else:
            #valid_move = list(env.board.legal_moves)
            #action = choose_move(valid_move)
            #env.step(str(action))
            #action = player2.choose_move(env)
            with tf.device("/gpu:2"):
                 action = player2.choose_move_mcts(env, player2)

#        print(action)
if __name__ == "__main__":
    training_episodes = 200000
    game_draw=White=Black=0
    cent_episode = []
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    white_path= "models/white_ckpts_at" + str(training_episodes) + "steps/"
    black_path ="models/black_ckpts_at" + str(training_episodes)+ "steps/"
    if not os.path.exists(white_path) :
        os.mkdir(white_path)
    if not os.path.exists(black_path) :
        os.mkdir(black_path)


    optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001)
    sess = tf.Session(config = config)
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
                                 discount_factor=0.8)
                                 discount_factor=0.8)'''
    with tf.device("/gpu:7"):
          b1=DQN_Bot(creat_network_3d, optimizer1, sess,black_path, block = 16 )
    with tf.device("/gpu:6"):     
          b2 =DQN_Bot(creat_network_3d, optimizer1,sess, white_path,  block = 32)
    env = chess_env.ChessEnv()
    for i_episode in range(training_episodes):
        env.reset()
        flip = random.randint(0, 1)
        if (flip == 0):
            with tf.device("/gpu:5"):
                play_bot_to_bot(env, b1,b2)
        else:
            with tf.device("/gpu:4"):
                play_dqn_to_mcts_dqn(env, b2,b1)


        if env.result == '1-0':
            print('White win')
            White +=1
        elif env.result == '0-1':
            print("Black Win") 
            Black +=1
        else:
            game_draw  +=1
            print("Match Draw") 

        if i_episode % 500 == 0:
            print("\n\n\nEpisode {}".format(i_episode))
            print("\twhite:" + str(White))
            print("\tblack:" + str(Black))
            print("\tdraw:" + str(game_draw))
            x =(White, Black, game_draw)
            cent_episode.append(x)
            game_draw=White=Black=0
            b1.save_ckpt(black_path, i_episode)
            b2.save_ckpt(white_path, i_episode)



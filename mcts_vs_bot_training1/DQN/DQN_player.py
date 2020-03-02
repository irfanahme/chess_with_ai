import tensorflow as tf
ver = tf.version.VERSION
if (int(ver[0])>1):
    print (True)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import os
from DQN.dqn_V import DeepQNetwork
#    from DQN.deep_q_network import DeepQNetwork
#    from DQN.dqn_gpu  import DeepQNetwork
from env.chess_env import all_input_planes
from env.move_func import make_move, available_move
from mcts import MCTSPlayer, evaluate

try:
    from DQN.dqn_V import DeepQNetwork
#    from DQN.deep_q_network import DeepQNetwork
#    from DQN.dqn_gpu  import DeepQNetwork   
    from env.chess_env import all_input_planes
    from env.move_func import make_move, available_move

except:
    #pass
    from dqn_V import DeepQNetwork
#    from dqn_gpu  import DeepQNetwork
#    from deep_q_network import DeepQNetwork 
    from chess_env import all_input_planes
    from move_func import make_move, available_move



class DQN_Bot():
    def __init__(self,  network, optimizer,sess, modelpath= "models/",  block=8):
        self.sess =sess 
        self.q_network = DeepQNetwork(self.sess,
                                 optimizer,
                                 network,
                                 block,
                                 state_dim = (18,8,8) ,
                                 num_actions = 1968,
                                 batch_size=2048,
                                 init_exp=0.6,         # initial exploration prob
                                 final_exp=0.01,   #0.1     # final exploration prob 
                                 anneal_steps=200000,  # N steps for annealing exploration
                                 double_q_learning=True,
                                 discount_factor=0.8)  # no need for discounting #0.8
        
        self.modelpath = modelpath
        self.var = self.q_network.var_lists
        self.saver = tf.train.Saver(self.var)
        self.restore()
    
    
    def choose_move_mcts(self, env, b1):
        state = all_input_planes(env.get_fen())
        b2 = MCTSPlayer(evaluate)
        action, probs = b2.get_action(env, b1)
        reward, gameover  = make_move(env, action)
       
        next_state = all_input_planes(env.get_fen())
        self.q_network.storeExperience(state , action, reward, next_state, gameover)
        self.q_network.updateModel()
        return action  
    
    def choose_move(self,  env):

        state = all_input_planes(env.get_fen())
        available_moves= available_move(list(env.board.legal_moves))
        action = self.q_network.eGreedyAction(state[np.newaxis,:], available_moves)
        reward, gameover  = make_move(env, action)
       
        next_state = all_input_planes(env.get_fen())
        self.q_network.storeExperience(state , action, reward, next_state, gameover)
        self.q_network.updateModel()
        return action 
    
    def get_p_v(self, state):
        #returms prob distribution and v_value
        return  self.q_network.get_p_V(state[np.newaxis,:])
    
    
    def save_ckpt(self, path = None, episode=0):
        if not path:
            path = self.modelpath
        if not os.path.exists(path) :
            os.mkdir(path)
        print(len(self.var), " variables are saved at episode  ", episode)

        self.saver.save(self.sess, path + "/" + str(episode))
    def set_saver(self, var) :
        self.saver =   tf.train.Saver(var)
#        returer
       
    
    def restore(self, path = None):
        if path==None:
            path = self.modelpath
        try:
            print("########", path)
            checkpoint = tf.train.get_checkpoint_state(path)
            print("$$$$$$$$$$ ", checkpoint)
            if checkpoint and checkpoint.model_checkpoint_path:
                print("$$$$$$$$$$$$$$$$$$$$$$$$$")
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                #self.saver.recover_last_checkpoints(path)
                print("\n\n\n\n\n*************  successfully loaded checkpoint*************\n\n\n\n\n")
        except: 
            print("\n\n\n\n\n*************couldnot load*************\n\n\n\n\n")
            


import tensorflow as tf
ver = tf.version.VERSION
if (int(ver[0])>1):
    print (True)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


import env.chess_env as chess_env
from DQN.net_function import creat_network_3d
from DQN.DQN_player import DQN_Bot
from env.move_func import make_move, available_move


optimizer1 = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001)

def human_choose_move(env):
    valid_move = available_move(list(env.board.legal_moves))
    print(valid_move)
    a = input("inter the move  :")
    make_move(env, str(a))
    print(env.board)

def play_bot_to_bot(env):
    b = DQN_Bot(creat_network_3d, optimizer1)
    b2 = DQN_Bot(creat_network_3d, optimizer2)
    
    while env.board.result() == "*":
        print(env.board)
        if env.white_to_move:
            action = b.choose_move(env)
        else:
            action = b2.choose_move(env)
        print(action)


def play_human_to_bot(env, p):
    b = DQN_Bot(creat_network_3d, optimizer1)
    if p == "WHITE":
        while env.board.result() == "*":
            print(env.board)
            if env.white_to_move:
                print("white player move")
                human_choose_move(env)
            else:
                action = b.choose_move(env)
                print(action)

    else:
        while env.board.result() == "*":
            print(env.board)
            if env.black_to_move:
                print("black player move")
                human_choose_move(env)
            else:
                action =b.choose_move(env)
                print(action)
    


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


if __name__ == "__main__":
    env = chess_env.ChessEnv()
    env.reset()
    def select():
        print("1:- Game play with human to human  : ")
        print("2:- Game play with Bot to Human  : ")
        print("3:- Game play with Bot to Bot  : ")
        check = int(input("Select number for play:  "))
        if check == 1:
            print("welcome to H vs H")
            p = input("Select Your Player   :")
            p = str.upper(p)
            player = ["WHITE", "BLACK"]
            if p == player[1]:
                env.change_turn(player[1])
                play_human_to_human(env, p)
            else:
                env.change_turn(player[0])
                play_human_to_human(env, p)
        elif check == 2:
            print("Welcome to B vs H")
            p = input("Select Your Player    :")
            p = str.upper(p)
            player = ["WHITE", "BLACK"]
            if p == player[1]:
#                env.change_turn(player[1])
                play_human_to_bot(env, p)
            else:
#                env.change_turn(player[0])
                play_human_to_bot(env ,p)
        elif check == 3:
            print("Welcome to B vs B")
            play_bot_to_bot(env)
            
        else:
            print("Plase select correct number")
            select()
    select()    





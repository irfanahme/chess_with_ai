import numpy as np
from collections import namedtuple, OrderedDict
import env.posible_moves as ps
import copy
from env.chess_env import all_input_planes
policy_softmax_temp = 1.0

def _softmax(x, softmax_temp):
    e_x = np.exp((x - np.max(x))/softmax_temp)
    return e_x / e_x.sum(axis=0)

def evaluate(leela_board , b):
        state = all_input_planes(leela_board.get_fen())
        policy, value = b.get_p_v(state)
        board = leela_board.board  #print(leela_board)
        return _evaluate(board, policy, value)
        
def _evaluate(board, policy, value):
    """This is separated from evaluate so that subclasses can evaluate based on raw policy/value"""
    if not isinstance(policy, np.ndarray):
        # Assume it's a torch tensor
        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
    # Results must be converted to float because operations
    # on numpy scalars can be very slow
    value = float(value[0])
    # Knight promotions are represented without a suffix in leela-chess
    # ==> the transformation is done in lcz_uci_to_idx
    legal_uci = [m.uci() for m in board.generate_legal_moves()]
    if legal_uci:
        legal_indexes = ps.possible_move_index(legal_uci)
        softmaxed = _softmax(policy[0][legal_indexes], policy_softmax_temp)
        softmaxed_aspython = map(float, softmaxed)
        policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed_aspython),
                                    key = lambda mp: (mp[1], mp[0]),
                                    reverse=True))
    else:
        policy_legal = OrderedDict()
    #value = value/2 + 0.5   # why?
    return policy_legal, value

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        #print(action_priors)
        for action, prob in action_priors.items():
            if action not in self._children:
                #print("&&&&&&&&&&&&&&action -------------", action)
                self._children[action] = TreeNode(self, prob)
            #else:
                #print("action is in  self._children:")
                #print("\n\n\n\n self._children", self._children )
                #print("\naction",action )


    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        #print("\n\n\n\n$$$$$$,self._children.items()\n\n",self._children.items())
        #print("^^^^^^",key=lambda act_node: act_node[1].get_value(c_puct))
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=10, n_playout=700):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state, b):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        #print(node)
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            #print(action, node)
            state.step(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state, b)
        #print('#########', action_probs, leaf_value)
        # Check for end of game.
        end, winner = state.game_end()
#        print("\n\n\n\n\n\^&^^^^^^^^^^^^^^^^^^^^^^^\n", state.board)
        #print("^&^^^^^^^^^^^^^^^^^^^^^^^", end)
        if not end:
            node.expand(action_probs)
        else:
             if winner == 0:    # tie
                 leaf_value = 0.0
             elif winner == -1:    # lose
                 leaf_value = -1
             if winner == 1:     #win
                 leaf_value = 1

            # for end state，return the "true" leaf_value
#            if winner == -1:  # tie
#               leaf_value = 0.0
#            else:
#                leaf_value = (
#                    1.0 if winner == state.get_current_player() else -1.0
#                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, b,  temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, b)

        # calc the move probabilities based on visit counts at the root node
        #print(self._root._children)
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        #print("act_visits\n\n\n\n\n\n\n\n\n\n", act_visits)
        if len(act_visits)!= 0:
            #print(len(act_visits))
            pass
        try:
            acts, visits = zip(*act_visits)
        except:
               #print("act_visits\n\n\n\n\n\n\n\n\n\n", act_visits)
               print(state_copy.board)
               print(state_copy.winner)
               print(self._root._children.item())
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=10, n_playout=700, is_selfplay=1):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, b, temp=1e-3, return_prob=1):
        sensible_moves = [m.uci() for m in board.board.generate_legal_moves()]
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        #print(sensible_moves)
        move_probs = list(np.zeros(10 *10))
        #print(move_probs)
        if len(sensible_moves) > 0:
            #print(board.board)
            acts, probs = self.mcts.get_move_probs(board, b, temp)
            #print(acts)
            #print(probs)
            for i in range (len(acts)):
                move_probs[i] = probs[i]
            #move_probs[i for i in range (len(acts))] = probs
        
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
                #print("if%%%%%")
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

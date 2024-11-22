"""
This file contains the MCTS implementation of a PlayerAgent
"""
from dgisim import PlayerAgent, GameState, Pid, PlayerAction, ActionGenerator, ActionType, Element, Cards, ActualDice, AbstractDice
from math import sqrt, log
import copy, random

class Node:
    def __init__(self, game_state: GameState, actions: list[PlayerAction], parent = None) -> None:
        # basic mcts
        self.state = copy.copy(game_state) # immutable
        self.num_wins = 0
        self.num_visits = 0
        self.children = [] # (action, child node)
        self.parent = parent

        # modify for tcg
        self.tried_actions = copy.copy(actions) # immutable, used to be untried_actions
        self.is_terminal = game_state.game_end() # gamestate is simulator


class OfflineAgent(PlayerAgent):
    BRANCH_LIMIT = 8
    ITERATION_BUDGET = 100
    
    def __init__(self, ) -> None:
        pass

    # mcts_search() 
    def choose_action(self, history: list[GameState], pid: Pid) -> PlayerAction:
        # moving intialization here
        self.game_state = history[-1]
        self.pid = pid
        self.root = Node(history[-1], [])

        iters = 0
        # mcts loop
        while (iters < self.ITERATION_BUDGET):
            if ((iters + 1) % 100 == 0):
                print("\riters/budget: {}/{}".format(iters + 1, self.ITERATION_BUDGET), end="")
            # select a node, rollout, and backpropagate
            node = self.select(self.root)
            winner = self.rollout(node)
            self.backpropagate(node, winner)
            iters += 1
        print()
        
        # return the best action, and the table of actions and their win values 
        _, action, _ = self.best_child(self.root, 0)
        return action

    def select(self, node: Node):
        # print("SELECTING...")
        while not node.is_terminal:
            # limit to X different actions
            # print(len(node.tried_actions))
            if len(node.tried_actions) <= self.BRANCH_LIMIT:
                # any unseen actions after the first X 
                # pretend they don't exist
                return self.expand(node)
            else:
                node = self.best_child(node, c=1)[0]
        return node

    def expand(self, node: Node):
        # duplicates allowed
        action = self._action_generator_chooser(node.state.action_generator(node.state.waiting_for()))
        node.tried_actions.append(action)
        self.game_state = copy.copy(node.state)
        self.game_state.action_step(node.state.waiting_for(), action) 
        # note: should it be taking an action step?
        child_node = Node(self.game_state, [], node)
        node.children.append((action, child_node))
        return child_node
    
    def best_child(self, node: Node, c: int = 1):
        best_child_node = None # to store the child node with best UCB
        best_action = None # to store the action that leads to the best child
        action_ucb_table = {} # to store the UCB values of each child node (for testing)

        N_parent = node.num_visits
        best_ucb = 0

        for child in node.children:
            N_child = child[1].num_visits
            Q_child = child[1].num_wins

            action_ucb_table[child[0]] = (Q_child / N_child) + (c * sqrt(2 * log(N_parent) / N_child))
            if action_ucb_table[child[0]] > best_ucb:
                best_ucb = action_ucb_table[child[0]]
                best_action = child[0]
                best_child_node = child[1]
        print()
        print(node.state)
        print(action_ucb_table)
        print()
        return best_child_node, best_action, action_ucb_table

    def rollout(self, node: Node):
        # try 3
        cur_state = copy.copy(self.game_state)
        while not cur_state.game_end():
            if cur_state.waiting_for() == None:
                cur_state = cur_state.step()
            else:
                action = self._action_generator_chooser(cur_state.action_generator(cur_state.waiting_for()))
                new_state = cur_state.action_step(cur_state.waiting_for(), action)
                if new_state == None:
                    print("error: new state is none")
                else:
                    cur_state = new_state
        # print("end rollout")
        # reward indicator for rollout
        reward = {}
        if cur_state.get_winner() == Pid.P1:
            reward[Pid.P1] = 1
            reward[Pid.P2] = 0
        elif cur_state.get_winner() == Pid.P2:
            reward[Pid.P1] = 0
            reward[Pid.P2] = 1
        else: # tie - both losers :3
            reward[Pid.P1] = 0
            reward[Pid.P2] = 0
        return reward
    
    def backpropagate(self, node: Node, result: list[Pid]):
        # each node should store the number of wins 
        # for the player of its **parent** node
        while (node is not None): 
            node.num_visits += 1
            node.num_wins += 1 - result[node.state.waiting_for()]
            node = node.parent

    def _action_generator_chooser(self, action_generator: ActionGenerator) -> PlayerAction:
        try:
            while not action_generator.filled():
                choices = action_generator.choices()
                # print(choices)
                if (type(choices) == tuple) :
                    choice = random.choice(choices)
                elif (type(choices) == ActualDice):
                    choice = action_generator.dice_available()
                elif (type(choices) == AbstractDice):
                    choice = action_generator.dice_available().smart_selection(choices)
                elif (type(choices) == Cards):
                    choice = choices.pick_random(random.randint(0,5))[0]
                action_generator = action_generator.choose(choice)
        except Exception as e:
            print(choice)
            print(f"Error with action_generator: {action_generator}")
            raise e
        return action_generator.generate_action()
    
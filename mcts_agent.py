"""
This file contains the MCTS implementation of a PlayerAgent
"""
from dgisim import PlayerAgent, GameState, Pid, PlayerAction, ActionGenerator, ActionType, Element, Cards, ActualDice, AbstractDice
from math import sqrt, log
import copy, random, time

class Node:
    def __init__(self, game_state: GameState, actions: list[PlayerAction], parent = None) -> None:
        # basic mcts
        self.state = copy.deepcopy(game_state)
        self.num_wins = 0
        self.num_visits = 0
        self.children = [] # (action, child node)
        self.parent = parent

        # modify for tcg
        self.tried_actions = copy.deepcopy(actions) # used to be untried_actions 
        self.is_terminal = game_state.game_end() # gamestate is simulator
        self.pid = game_state.active_player_id


class MCTSAgent(PlayerAgent):
    def __init__(self, history: list[GameState], pid: Pid) -> None:
        self.budget = 20
        self.simulator = history[-1]
        # self.act_gen = self.simulator.action_generator(pid)
        self.pid = pid
        self.root = Node(history[-1], [])

    # mcts_search() 
    def choose_action(self, history: list[GameState], pid: Pid) -> PlayerAction:
        iters = 0
        # mcts loop
        while (iters < self.budget):
            if ((iters + 1) % 100 == 0):
                print("\riters/budget: {}/{}".format(iters + 1, self.budget), end="")
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
        while not node.is_terminal:
            # limit to X different actions
            if len(node.tried_actions) <= 5:
                # any unseen actions after the first X 
                # pretend they don't exist
                return self.expand(node)
            else:
                node = self.best_child(node, c=1)[0]
        return node

    def expand(self, node: Node):
        # print("!!!!!", node.state)
        # duplicates allowed
        action = self._action_generator_chooser(node.state.action_generator(node.pid))
        # print(action)
        node.tried_actions.append(action)
        self.simulator = copy.deepcopy(node.state)
        self.simulator.action_step(node.pid, action)
        child_node = Node(self.simulator, [], node)
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

        return best_child_node, best_action, action_ucb_table

    def backpropagate(self, node: Node, result: list[int]):
        # each node should store the number of wins 
        # for the player of its **parent** node
        while (node is not None): 
            node.num_visits += 1
            node.num_wins += 1 - result[node.state[0]]
            node = node.parent

    def rollout(self, node: Node):
        # rollout (random)
        cur_state = copy.deepcopy(node.state)
        count = 0
        while not cur_state.game_end():
            print(cur_state.active_player_id, cur_state.waiting_for())
            if count < 5:
                print("rollout: ", cur_state.active_player_id, cur_state)
            if cur_state.waiting_for() == None:
                cur_state.step()
            else:
                action = self._action_generator_chooser(cur_state.action_generator(cur_state.active_player_id))
                cur_state.action_step(cur_state.active_player_id, action)
            
            print(cur_state.active_player_id, action, count)
            count += 1
        time.sleep(5)
        print("rollout finished")
        

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
    
    def _action_generator_chooser(self, action_generator: ActionGenerator) -> PlayerAction:
        try:
            while not action_generator.filled():
                # print("not filled!")
                choices = action_generator.choices()
                choice: DecidedChoiceType  # type: ignore
                if isinstance(choices, tuple):
                    game_state = action_generator.game_state
                    if game_state.phase == game_state.mode.roll_phase() and random.random() < 0.8:
                        choices = tuple(c for c in choices if c is not ActionType.END_ROUND)
                    choice = random.choice(choices)
                    # print("tuple", choice)
                    action_generator = action_generator.choose(choice)
                elif isinstance(choices, AbstractDice):
                    optional_choice = action_generator.dice_available().smart_selection(
                        choices,
                        action_generator.game_state.get_player(action_generator.pid).characters,
                    )
                    if optional_choice is None:
                        raise Exception(f"There's not enough dice for {choices} from "  # pragma: no cover
                                        + f"{action_generator.dice_available()} at game_state:"
                                        + f"{action_generator.game_state}")
                    choice = optional_choice
                    # print("abstract", choice)
                    action_generator = action_generator.choose(choice)
                elif isinstance(choices, Cards):
                    _, choice = choices.pick_random(random.randint(0, choices.num_cards()))
                    # print("cards", choice)
                    action_generator = action_generator.choose(choice)
                elif isinstance(choices, ActualDice):
                    game_state = action_generator.game_state
                    wanted_elems = game_state.get_player(
                        action_generator.pid
                    ).characters.all_elems()
                    if game_state.phase == game_state.mode.roll_phase():
                        choice = ActualDice(dict(
                            (elem, choices[elem])
                            for elem in choices.elems()
                            if not (elem is Element.OMNI or elem in wanted_elems)
                        ))
                    else:
                        _, choice = choices.pick_random_dice(random.randint(0, choices.num_dice()))
                    # print("actual", choice)
                    action_generator = action_generator.choose(choice)
                else:
                    raise NotImplementedError
        except Exception as e:
            print(f"Error with action_generator: {action_generator}")
            raise e
        action = action_generator.generate_action()
        # print(action)
        return action
    
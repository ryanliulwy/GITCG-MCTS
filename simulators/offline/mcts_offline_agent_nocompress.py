"""
This file contains the MCTS implementation of a PlayerAgent
"""
from dgisim import PlayerAgent, GameState, Pid, PlayerAction, Instruction, ActionGenerator, ActionType, Element, Cards, ActualDice, AbstractDice
from dgisim import CardAction, CardsSelectAction, CharacterSelectAction, DeathSwapAction, DiceSelectAction, ElementalTuningAction, EndRoundAction, SkillAction, SwapAction
from math import sqrt, log
import copy, random, json, os
    
class Node:
    def __init__(self, game_state: GameState, actions: list[PlayerAction], pid, parent = None) -> None:
        # basic mcts
        self.state = copy.copy(game_state) # immutable
        self.num_wins = 0
        self.num_visits = 0
        self.children = [] # (action, child node)
        self.parent = parent
        self.pid = pid

        # modify for tcg
        self.tried_actions = copy.copy(actions) # immutable, used to be untried_actions
        self.is_terminal = game_state.game_end() # gamestate is simulator

class CompressedNode:
    # local vars: 
    # num_dice, num_cards, active_char, energy_tier, hp_bucket

    # TODO: below
    # --------- check one character or all?
    # card “types”? (offensive, defensive, etc…?)
    # only represent opposing character “roles”? (damage, support, etc.)
    # ignore less impactful status effects (like “full”)

    def __init__(self, node: Node):
        # examine only the generalized game state
        # independent of the history
        state = node.state.get_player(node.pid)

        # current game phase
        self.game_phase = node.state.phase.dict_str()
        # available dice → number of dice (potential: check effective dice)
        self.num_dice = sum(state.dice.readonly_dice_collection_ordered().values())
        # available cards → number of cards
        self.num_cards = state.hand_cards.num_cards()
        # ids of alive characters
        self.alive_characters = [c.get_id() for c in state.characters.get_alive_characters()]
        # active character
        self.active_char = state.characters.get_active_character() # sometimes None 
        if (self.active_char != None):
            # energy tiers (empty, partial, full)
            energy_percentage = self.active_char.energy / self.active_char.max_energy
            if (energy_percentage == 0): self.energy_tier = "empty"
            elif (energy_percentage == 1): self.energy_tier = "full"
            else: self.energy_tier = "partial"
            # HP buckets (high > 70%, medium 40-70%, low < 40%)
            hp_percentage = self.active_char.hp / self.active_char.max_hp
            if (hp_percentage > 0.7): self.hp_bucket = "high"
            elif (hp_percentage >= 0.4): self.hp_bucket = "medium"
            else: self.hp_bucket = "low"
            # set character to name only (not object)
            self.active_char = self.active_char.name() 
        else:
            self.energy_tier = None
            self.hp_bucket = None

    def __eq__(self, other):
        if (self.game_phase != other.game_phase): return False
        if (self.num_dice != other.num_dice): return False
        if (self.num_cards != other.num_cards): return False
        if (self.alive_characters != other.alive_characters): return False
        if (self.active_char != other.active_char): return False
        if (self.energy_tier != other.energy_tier): return False
        if (self.hp_bucket != other.hp_bucket): return False
        return True

    def __hash__(self):
        # hash all __eq__ attributes
        return hash((self.num_dice, self.num_cards, self.active_char, self.energy_tier, self.hp_bucket))
    
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False)
    
    def toJSONPretty(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False,
            indent=4)

# custom JSON encoder
# https://dottore-genius-invokation-tcg-simulator.readthedocs.io/en/stable/action/player-action-n-instruction.html
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # print(type(obj))
        if isinstance(obj, CardAction):
            return {
                "action_type": "CardAction",
                "card": obj.card.name()
            }
        if isinstance(obj, CardsSelectAction):
            return {
                "action_type": "CardsSelectAction",
                "selected_cards": {card.name(): count for card, count in obj.selected_cards._cards.items()}
            }
        if isinstance(obj, CharacterSelectAction):
            return {
                "action_type": "CharacterSelectAction",
                "char_id": obj.char_id
                # "character": get_character(obj.char_id).name(),
            }
        if isinstance(obj, DeathSwapAction):
            return {
                "action_type": "DeathSwapAction",
                "char_id": obj.char_id
            }
        if isinstance(obj, DiceSelectAction):
            return {
                "action_type": "DiceSelectAction"
                #"selected_dice": {element: count for element, count in obj.selected_dice.dice_ordered.items()}
            }
        if isinstance(obj, ElementalTuningAction):
            return {
                "action_type": "ElementalTuningAction",
                "card": obj.card.name()
            }
        if isinstance(obj, EndRoundAction):
            return {
                "action_type": "EndRoundAction"
            }
        if isinstance(obj, SkillAction):
            return {
                "action_type": "SkillAction",
                # "skill": obj.skill 
                # SkillAction(skill=SKILL2, instruction=DiceOnlyInstruction(dice={ELECTRO: 2, OMNI: 1}))
                # TypeError: Object of type CharacterSkill is not JSON serializable

            }
        if isinstance(obj, SwapAction):
            return {
                "action_type": "SwapAction",
                "char_id": obj.char_id
            }
        return super().default(obj)

class OfflineAgent(PlayerAgent):
    BRANCH_LIMIT = 8
    ITERATION_BUDGET = 100
    
    def __init__(self, TRAIN_NAME: str = None, offline_dict: dict = {}) -> None:
        self.TRAIN_NAME = TRAIN_NAME
        # opening save file for training
        self.SAVE_FILE = 'offline_save_file_' + TRAIN_NAME + '.txt'
        # create the file if it doesn't exist
        # if not os.path.exists(self.SAVE_FILE):
        #     with open(self.SAVE_FILE, 'a') as f:
        #         pass  
        # # read the file 
        self.offline_dict = offline_dict
        # try:
        #     with open(self.SAVE_FILE, 'r') as f:
        #         # TODO debug this section
        #         data = f.read().strip()
        #         # print("data: ", json.loads(data))
        #         # self.offline_dict = {json.loads(k): json.loads(v) for k, v in data.items()} if data else {}  
        #         self.offline_dict = json.loads(data) if data else {}  
        # except json.JSONDecodeError as e:
        #     print(f"ERROR: JSON decode error: {e}")
        #     self.offline_dict = {}
        # except Exception as e:
        #     print(f"ERROR: unexpected exception: {e}")
        #     self.offline_dict = {}
        # except:
        #     print("ERROR: exception")
        #     self.offline_dict = {}
        # print(self.offline_dict)
        # offline_dict is { gamestate (compressednode) : action }

        
    # mcts_search() 
    def choose_action(self, history: list[GameState], pid: Pid) -> PlayerAction:
        # moving intialization here
        self.game_state = history[-1]
        self.pid = pid
        self.root = Node(history[-1], [], pid)

        iters = 0
        # mcts loop
        while (iters < self.ITERATION_BUDGET):
            if ((iters + 1) % 50 == 0):
                print("\riters/budget: {}/{}".format(iters + 1, self.ITERATION_BUDGET), end="")
            # select a node, rollout, and backpropagate
            node = self.select(self.root)
            print(iters, CompressedNode(node).toJSON())
            winner = self.rollout(node)
            self.backpropagate(node, winner)
            iters += 1
        print()

        # write to offline after 1 iteration cycle

        # # print()
        # print("SAVING...")
        # # print (self.offline_dict)
        # with open(self.SAVE_FILE, 'w') as f:
        #     f.write(json.dumps(self.offline_dict, sort_keys=False, indent=4))
        #     # json.dump(self.offline_dict, f)
        # # print()
        
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
                # self.last_choice = 'expansion'
                return self.expand(node)
            else:
                compressed_node = repr(CompressedNode(node).toJSON())
                compressed_node = compressed_node[1:-1]
                if compressed_node.replace('"', "'") in self.offline_dict: # best action exists already
                    action = self.offline_dict[compressed_node.replace('"', "'")] 
                    game_state = copy.copy(node.state)
                    #print("game state:", game_state)
                    #print("node.state:", node.state)
                    #print("game phase:", game_state.phase.dict_str())
                    game_state.action_step(node.state.waiting_for(), action) 
                    node = Node(game_state, [], self.pid, node)
                    print("used dict! :3")
                else:
                    node, best_action, _ = self.best_child(node, c=1)
                    # self.last_choice = 'best child'
                    compressed_node = repr(CompressedNode(node).toJSON())
                    compressed_node = compressed_node[1:-1]
                    self.offline_dict[compressed_node.replace('"', "'")] = best_action # uncompressed action
                # self.offline_dict[compressed_node.replace('"', "'")] = json.dumps(best_action, cls=CustomJSONEncoder).replace('"', "'")  # update for offline
                # self.offline_dict[CompressedNode(node).toJSON()] = best_action # update for offline
        return node


    def expand(self, node: Node):
        # duplicates allowed
        action = self._action_generator_chooser(node.state.action_generator(node.state.waiting_for()))
        node.tried_actions.append(action)
        self.game_state = copy.copy(node.state)
        self.game_state.action_step(node.state.waiting_for(), action) 
        # note: should it be taking an action step?
        child_node = Node(self.game_state, [], self.pid, node)
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
        # print()
        # print(node.state)
        # print(action_ucb_table)
        # print()
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
    
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np

'''
Double Mini GITCG Setup:
- All Omni Dice
- 1 Character Only (Kaeya)
- Search Space: ~10^5 (similar to tic-tac-toe)
'''

class GITCGDoubleMiniGymEnv(gym.Env):
    # setup
    actions = {
        "kaeya_normal": { "dmg": 2, "dice_cost": 3, "energy": 1},
        "kaeya_skill": { "dmg": 3, "dice_cost": 3, "energy": 1},
        "kaeya_burst": {"dmg": 3, "dice_cost": 3, "energy_cost": 2},
        "broken_rimes_echo": { "atk_discount": 1, "dice_cost": 2, "card_type": "artifact"},
        "hash_brown": { "hp": 2, "dice_cost": 1, "card_type": "food" },
        "sweet_madame": { "hp": 1, "dice_cost": 0, "card_type": "food" },
        "skyward_blade": { "atk_permanent": 1, "atk_per_turn": 1, "dice_cost": 3, "card_type": "weapon"},
        "end_round_action": { "dice_cost": 0}
    }
    dice_per_turn = 4
    turn = 1

    def __init__(self):
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self.observations = {}
        
        self.vocab = ["broken_rimes_echo", "hash_brown", "sweet_madame", "skyward_blade", "kaeya_normal", "kaeya_skill", "kaeya_burst", "end_round_action"]
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab, start=1)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        self.reset()
    
    def observation_space(self): 
        return {
            agent: spaces.Dict({
                "Kaeya": spaces.Dict({
                    "max_hp": spaces.Discrete(10),
                    "hp": spaces.Discrete(10),
                    "max_energy": spaces.Discrete(2),
                    "energy": spaces.Discrete(2),
                    "atk_permanent": spaces.Discrete(5),
                    "atk_per_turn": spaces.Sequence(  
                            spaces.Tuple((spaces.Discrete(5), spaces.Discrete(2)))  
                        ),
                    "atk_discount": spaces.Discrete(5),
                    "actions": spaces.MultiDiscrete([len(self.vocab)+1] * 3), # normal / skill / burst
                    "artifact": spaces.Text(15),
                    "weapon": spaces.Text(15),
                    "full": spaces.Discrete(2)
                }),
                "dice": spaces.Discrete(10),
                "cards": spaces.MultiDiscrete([len(self.vocab)+1] * 5), # 5 cards in hand to start
                "declared_end": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(len(self.actions))
            })
            for agent in self.possible_agents
        }
    

    def action_space(self):
        return spaces.Discrete(len(self.actions))

    def reset(self, seed=None, options=None):
        self.observations = {
            agent: {
                "Kaeya": {
                    "max_hp": 10,
                    "hp": 10,
                    "max_energy": 2,
                    "energy": 0,
                    "atk_permanent": 0, # bonus atk 
                    "atk_per_turn": [], # bonus atk once per turn
                    "atk_discount": 0,
                    "actions": np.array([
                            self.word_to_id["kaeya_normal"],
                            self.word_to_id["kaeya_skill"],
                            self.word_to_id["kaeya_burst"]
                        ], dtype=np.int8),
                    "artifact": "", # None
                    "weapon": "", # None
                    "full": 0 # False
                },
                "dice": self.dice_per_turn,
                "cards": np.array([
                    self.word_to_id["broken_rimes_echo"],
                    self.word_to_id["hash_brown"],
                    self.word_to_id["sweet_madame"],
                    self.word_to_id["sweet_madame"],
                    self.word_to_id["skyward_blade"]
                ], dtype=np.int8),
                "declared_end": int(False),
                "action_mask": np.array([1] * len(self.actions), dtype=np.int8)
            }
            for agent in self.possible_agents
        }
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.done = False
        return self.observations, {}

    def get_action_mask(self, agent):
        # valid - 1, invalid - 0
        mask = [0] * len(self.actions)
        for action in self.actions.keys():
            value = self.actions[action] 
            if self.observations[agent]["declared_end"] and action != "end_round_action":
                continue # need to end round after end
            if "card_type" in action and self.word_to_id[action] not in self.observations[agent]["cards"]:
                continue  # card not in hand
            if value["dice_cost"] - self.observations[agent]["Kaeya"]["atk_discount"] > self.observations[agent]["dice"]:
                continue  # not enough dice
            if action.__contains__("burst") and self.observations[agent]["Kaeya"]["max_energy"] != self.observations[agent]["Kaeya"]["energy"]:
                continue # not enough energy
            if "hp" in action and self.observations[agent]["Kaeya"]["full"]:
                continue # full, can't eat more
            mask[list(self.actions).index(action)] = 1 # valid otherwise
        return mask

    
    def step(self, action):
        agent = self.agent_selection
        other_agent = "player_0" if agent == "player_1" else "player_1"
        reward = 0
        total_dmg = 0
        action = self.id_to_word[action+1]

        # check if action is valid?
        if self.get_action_mask(agent)[list(self.actions).index(action)] == 0:
            reward = -100 # invalid
            action = "end_round_action"
        
        # apply action
        if action == "end_round_action":
            self.observations[agent]["declared_end"] = True
        elif "card_type" not in self.actions[action]: # character atk
            atk = self.actions[action]["dmg"]
            atk += self.observations[agent]["Kaeya"]["atk_permanent"] 
            for i in range(len(self.observations[agent]["Kaeya"]["atk_per_turn"])):
                bonus = self.observations[agent]["Kaeya"]["atk_per_turn"][i][0]
                used = self.observations[agent]["Kaeya"]["atk_per_turn"][i][1]
                # print(bonus, used)
                if used == False:
                    self.observations[agent]["Kaeya"]["atk_per_turn"][i] = (bonus, True)
                    atk += bonus
            total_dmg = atk
            # add energy
            if "energy" in self.actions[action]:
                self.observations[agent]["Kaeya"]["energy"] += self.actions[action]["energy"]
                self.observations[agent]["Kaeya"]["energy"] = max(self.observations[agent]["Kaeya"]["energy"], self.observations[agent]["Kaeya"]["max_energy"])
            elif self.observations[agent]["Kaeya"]["energy"] == self.observations[agent]["Kaeya"]["max_energy"]:
                self.observations[agent]["Kaeya"]["energy"] = 0 # use burst
            else:
                pass # can't use burst, shouldn't happen
            # deal dmg to opposite character 
            self.observations[other_agent]["Kaeya"]["hp"] -= atk
        else:
            np.delete(self.observations[agent]["cards"], [self.word_to_id[action]])
            if "hp" in self.actions[action]:
                cur_hp = self.observations[agent]["Kaeya"]["hp"]
                cur_hp += self.actions[action]["hp"]
                max_hp = self.observations[agent]["Kaeya"]["max_hp"]
                self.observations[agent]["Kaeya"]["hp"] = min(max_hp, cur_hp)
                self.observations[agent]["Kaeya"]["full"] = True
            if "atk_permanent" in self.actions[action]:
                self.observations[agent]["Kaeya"]["atk_permanent"] += self.actions[action]["atk_permanent"]
            if "atk_per_turn" in self.actions[action]:
                self.observations[agent]["Kaeya"]["atk_per_turn"].append((self.actions[action]["atk_per_turn"], False)) # once per turn
            if self.actions[action]["card_type"] == "artifact":
                self.observations[agent]["Kaeya"]["artifact"] = action
            if self.actions[action]["card_type"] == "weapon":
                self.observations[agent]["Kaeya"]["weapon"] = action

        # rewards for good actions
        if action in ["kaeya_normal", "kaeya_skill", "kaeya_burst"]:
            reward += total_dmg + self.observations[agent]["Kaeya"]["hp"] # add current hp to incentivize rounds that end with higher hp
        
        # kills opponent
        if self.observations[other_agent]["Kaeya"]["hp"] <= 0:
            reward += 100 #big reward
            # directly end game here in mini double
            self.done = True 
            self.winner = agent

        # check for new round
        if self.observations[agent]["declared_end"] and self.observations[other_agent]["declared_end"]:
            self.observations[agent]["Kaeya"]["atk_per_turn"][1] = False
            self.observations[agent]["declared_end"] = False
            self.observations[agent]["full"] = False
            self.observations[agent]["dice"] = 4
            
            self.observations[other_agent]["declared_end"] = False
            self.observations[other_agent]["declared_end"] = False
            self.observations[other_agent]["full"] = False
            self.observations[other_agent]["dice"] = 4

            turn += 1
        return self.observations, reward, self.done, False, {}


    def close(self):
        pass
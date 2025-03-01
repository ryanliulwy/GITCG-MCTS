"""
This file runs 1000 games with the default random agent to test the speed of the simulator.
"""
import dgisim as dg
from dgisim import GameState, Pid
from dgisim.agents import RandomAgent
# from custom_agents.mcts_agent import MCTSAgent
from mcts_offline_agent import OfflineAgent
import random, time

from deck_aoe import deck1 as deckAOE # buggy
from deck_single import deck as deckSingle # untested
from deck_default import deck1 as deckDefault

import pickle # offline pls

TRAIN_NAME = 'default'

try:
    with open('offline_nocompress_dict.pkl', 'rb') as f:
        offline_dict = pickle.load(f)
    print("loaded existing model")
except FileNotFoundError:
    print("using new model")
    offline_dict = {}

agRand = RandomAgent()
agMCTS = OfflineAgent(TRAIN_NAME, offline_dict)

start = time.time()

p1_wins = 0
p2_wins = 0
total_games = 1
for i in range(total_games):
    game_state = GameState.from_decks(
        mode=dg.AllOmniMode(),
        p1_deck=deckDefault,
        p2_deck=deckDefault
    ) # always use the same starting decks 
    history = [game_state]
    while not game_state.game_end():
        if game_state.waiting_for() is None:
            game_state = game_state.step()
        elif game_state.waiting_for() is Pid.P1:
            current_player = game_state.waiting_for()
            action = agRand.choose_action(history, current_player)
            game_state = game_state.action_step(current_player, action)
            print(current_player, action)
        elif game_state.waiting_for() is Pid.P2: # MCTS goes second
            current_player = game_state.waiting_for()
            action = agMCTS.choose_action(history, current_player)
            game_state = game_state.action_step(current_player, action)
            print(current_player, action)
        history.append(game_state)
    print("Game #%d complete" % i)
    print(game_state.get_winner())
    if game_state.get_winner() == Pid.P1:
        p1_wins += 1
    elif game_state.get_winner() == Pid.P2:
        p2_wins += 1

with open('offline_no_compress_dict.pkl', 'wb') as f:
    pickle.dump(agMCTS.offline_dict, f)

end = time.time()
print(end - start)
print("P1 wins:", p1_wins)
print("P2 wins:", p2_wins)
print("Totaal games:", total_games)
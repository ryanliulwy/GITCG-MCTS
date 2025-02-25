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
import wandb

TRAIN_NAME = 'default'

try:
    with open('offline_dict.pkl', 'rb') as f:
        offline_dict = pickle.load(f)
    print("loaded existing model")
except FileNotFoundError:
    print("using new model")
    offline_dict = {}

agRand = RandomAgent()
agMCTS = OfflineAgent(TRAIN_NAME, offline_dict)

wandb.init(project="dgisim-offline-mcts", name=f"random_vs_o-mcts_50")
p1_wins = 0
p2_wins = 0
total_games = 30

action_counts = {} # action tracker

wandb.config.update({
    "total_games": total_games,
    "p1_agent": "RandomAgent",
    "p2_agent": "O-MCTSAgent"
})

start = time.time()

p1_wins = 0
p2_wins = 0
total_games = 20
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
            # tracking
            action_type = action.__class__.__name__  # e.g., "CardAction", "SkillAction"
            key = f"{current_player} {action_type}"
            if key in action_counts:
                action_counts[key] += 1  
            else:
                action_counts[key] = 1
        elif game_state.waiting_for() is Pid.P2: # MCTS goes second
            current_player = game_state.waiting_for()
            action = agMCTS.choose_action(history, current_player)
            game_state = game_state.action_step(current_player, action)
            print(current_player, action)
            # tracking
            action_type = action.__class__.__name__  # e.g., "CardAction", "SkillAction"
            key = f"{current_player} {action_type}"
            if key in action_counts:
                action_counts[key] += 1  
            else:
                action_counts[key] = 1
        history.append(game_state)
    print("Game #%d complete" % i)
    print("Winner is", game_state.get_winner())
    if game_state.get_winner() == Pid.P1:
        p1_wins += 1
    elif game_state.get_winner() == Pid.P2:
        p2_wins += 1
    print(action_counts)

with open('offline_dict.pkl', 'wb') as f:
    pickle.dump(agMCTS.offline_dict, f)

end = time.time()
print(end - start, " seconds taken.")
print("P1 wins:", p1_wins)
print("P2 wins:", p2_wins)
print("Total games:", total_games)


wandb.log({
    "total_duration": end - start,
    "final_p1_wins": p1_wins,
    "final_p2_wins": p2_wins,
    "final_p1_win_rate": p1_wins / total_games,
    "final_p2_win_rate": p2_wins / total_games
})

wandb.log(action_counts)
for action, count in action_counts.items():
    print(f"{action}: {count} times")

wandb.finish()
"""
This file runs 1000 games with the default random agent to test the speed of the simulator.
"""

from dgisim import LinearEnv, ActionGenerator, GameState, Pid, ActualDice, AbstractDice, Cards, PlayerAgent
from dgisim.agents import RandomAgent
from custom_agents.mcts_agent import MCTSAgent
from mcts_offline_agent import OfflineAgent
import random, time

env = LinearEnv()
agRand = RandomAgent()

start = time.time()

p1_wins = 0
p2_wins = 0
total_games = 1
for i in range(total_games):
    env.reset()
    game_state, encoded_state, reward, turn, done = env.view()
    pid = Pid.P1
    history = [game_state]

    agMCTS = MCTSAgent(history, pid)

    while not done:
        if (turn == 2):
            pid = Pid.P2
            ag = agRand
        else:
            pid = Pid.P1
            ag = agMCTS
        action = ag.choose_action(history, pid)
        game_state, encoded_state, reward, turn, done = env.step(action)
        history.append(game_state)
        if (game_state.game_end()):
            print("Game #%d complete" % (i+1))
            if game_state.get_winner() == Pid.P1:
                p1_wins += 1
            elif game_state.get_winner() == Pid.P2:
                p2_wins += 1
            print(game_state.get_winner())

end = time.time()
print(end - start)
print("P1 wins:", p1_wins)
print("P2 wins:", p2_wins)
print("Total games:", total_games)
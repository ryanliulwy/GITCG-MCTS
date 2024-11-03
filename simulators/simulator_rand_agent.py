"""
This file runs 1000 games with the default random agent to test the speed of the simulator.
"""

from dgisim import LinearEnv, ActionGenerator, GameState, Pid, ActualDice, AbstractDice, Cards, PlayerAgent
from dgisim.agents import RandomAgent
import random, time

env = LinearEnv()
ag = RandomAgent()

start = time.time()

for i in range(1000):
    env.reset()
    game_state, encoded_state, reward, turn, done = env.view()
    pid = Pid.P1
    history = [game_state]

    while not done:
        if (turn == 2): pid = Pid.P2
        else: pid = Pid.P1
        action = ag.choose_action(history, pid)
        game_state, encoded_state, reward, turn, done = env.step(action)
        history.append(game_state)
        if (game_state.game_end()): print ("Game #%d complete" % i)

end = time.time()
print(end - start)
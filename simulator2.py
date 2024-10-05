from dgisim import LinearEnv, ActionGenerator, GameState, Pid, ActualDice, AbstractDice, Cards
import random

env = LinearEnv()

for _ in range(1):
    env.reset()
    game_state, encoded_state, reward, turn, done = env.view()
    pid = Pid.P1

    while not done:
        act_gen: ActionGenerator = game_state.action_generator(pid)
        while not act_gen.filled():
            choices = act_gen.choices()
            if (type(choices) == tuple) :
                choice = random.choice(choices)
            elif (type(choices) == ActualDice):
                choice = random.choice(list(choices._dice.keys()))
            elif (type(choices) == AbstractDice):
                choice = random.choice(list(choices.keys()))
            elif (type(choices) == Cards):
                choice = choices.pick_random(random.randint(0,5))[0]
            act_gen = act_gen.choose(choice)
        action = act_gen.generate_action()
        game_state, encoded_state, reward, turn, done = env.step(action)
        if (pid == Pid.P1): pid = Pid.P2
        else: pid = Pid.P1
        if (game_state.game_end()): print (game_state.get_winner())
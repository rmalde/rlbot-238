import os
from stable_baselines3 import PPO


class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        attemptName = "TwoKickoff" #Change for attempt
        
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        attemptPath = os.path.join(cur_dir, '../../models/'+ attemptName)
        model = PPO.load(attemptPath)
        self.model = model

    def act(self, state):
        action, _state = self.model.predict(state)
        return action

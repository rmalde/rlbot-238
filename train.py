import rlgym
import sys
from stable_baselines3 import PPO
from customObsBuilder import BaseObsBuilder
from customRewardFunction import BaseRewardFunction
from customTerminalCondition import BaseTerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
import pickle

def get_env():
    reward_fn = BaseRewardFunction()
    #reward_fn = LiuDistancePlayerToBallReward()
    obs_builder = BaseObsBuilder()
    terminal_condition = BaseTerminalCondition()
    timeout_condition = TimeoutCondition(50) #Times put after certain amount of time, I think 240=10sec? 

    env = rlgym.make(
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        terminal_conditions=[terminal_condition, timeout_condition]
    )
    return env


def train():
    env = get_env()
    #model = PPO("MlpPolicy", env=env, verbose=1)
    attemptName = "DistaceToBall" #Change for attempt
    model = PPO.load(attemptName, env=env)
    try:
        model.learn(total_timesteps=int(1e4))

        model.save(attemptName)
        print("Saved Model")

    except KeyboardInterrupt: #if we ctrl+c, save what we've trained so far (not sure if this will actually work but it's worth a try)
        #with open('model.p', 'wb') as handle:
            #pickle.dump(model, handle)
        model.save(attemptName)
        print("Sucessfully Interrupted: Saved Model")

if __name__ == "__main__":
    #add any parameters here, sys.argv[]
    train()

import rlgym
import sys
from stable_baselines3 import PPO
from customObsBuilder import BaseObsBuilder
from customRewardFunction import BaseRewardFunction
from customRewardFunction import combReward
from customTerminalCondition import BaseTerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.obs_builders import AdvancedObs
import pickle

def get_env():
    #reward_fn = combReward
    # reward_fn = BaseRewardFunction()
    # reward_fn = LiuDistancePlayerToBallReward()
    reward_fn = BaseRewardFunction()
    # obs_builder = BaseObsBuilder()
    obs_builder = AdvancedObs()
    terminal_condition = BaseTerminalCondition()
    timeout_condition = TimeoutCondition(250) #Times put after certain amount of time, I think 240=10sec? 

    env = rlgym.make(
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        terminal_conditions=[terminal_condition, timeout_condition]
    )
    return env


def train(isAnUpdate, update, new):
    env = get_env()
    model = None
    if isAnUpdate:
        model = PPO.load("models/" + update, env=env)
    else:
        model = PPO("MlpPolicy", env=env, verbose=1)

    try:
        model.learn(total_timesteps=int(1e4))

        model.save("models/" + new)
        print("Saved Model")

    except KeyboardInterrupt: #if we ctrl+c, save what we've trained so far (not sure if this will actually work but it's worth a try)
        #with open('model.p', 'wb') as handle:
            #pickle.dump(model, handle)
        model.save("models/" + new)
        print("Sucessfully Interrupted: Saved Model")

if __name__ == "__main__":
    update = ''
    new = ''
    isAnUpdate = False
    try:
        if sys.argv[1] == 'new':
            new = sys.argv[2]
            print(f'Saving new model to models/{new}')
        elif sys.argv[1] == 'update':
            isAnUpdate = True
            update = sys.argv[2]
            new = sys.argv[3]
            print(f'Loading model {update}')
            print(f'Saving new model to models/{new}')
        else:
            raise Exception("Didn't find new or update")
    except Exception as e:
        print("""Error: invalid inputs.
        Usage: 
        train.py new <save_model_name>
        train.py update <load_model_name> <save_model_name>
        """)
    train(isAnUpdate, update, new)

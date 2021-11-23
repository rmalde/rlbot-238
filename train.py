import rlgym
import sys
from customObsBuilder import BaseObsBuilder
from customRewardFunction import BaseRewardFunction
from customTerminalCondition import BaseTerminalCondition

def get_env():
    reward_fn = BaseRewardFunction()
    obs_builder = BaseObsBuilder()
    terminal_condition = BaseTerminalCondition()

    env = 



def train():
env = rlgym.make() #change this for custom reward, obs builder, terminal cond
    model = PPO("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=int(1e6))

if __name__ == "__main__":
    #add any parameters here, sys.argv[]
    train()

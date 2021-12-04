# from stable_baselines3 import PPO
# obs = [-1.28890869e+00, 1.08534778e-01, 4.05000007e-02, -5.96661271e-01, 4.85699994e-02, 0.00000000e+00, -1.23284601e-01, -1.71811253e+00, 8.24871397e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, -5.36717423e-01, -1.25629560e+00, 3.31043484e-02, -1.39769552e-01, -6.62078247e-01, -9.17391285e-05, -7.52191268e-01, 1.36483038e+00, 7.39565227e-03, -5.99503904e-01, 8.00314664e-01, -9.56595791e-03, -5.72767886e-03, 7.66167941e-03, 9.99954245e-01, -4.56891718e-01, 7.10648246e-01, 9.17391285e-05, 0.00000000e+00, -1.30507047e-04, 2.19952131e-03, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00]
# model = PPO.load("models/AllKickoffsv1")
# for _ in range(10):
#     action, _states = model.predict(obs, deterministic=True)
#     print(action)
#     print(_states)
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import BallTouchedCondition, TimeoutCondition
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from customStateSetter import CustomStateSetter
from stable_baselines3 import PPO


env = rlgym.make(reward_fn=EventReward(goal=1.), obs_builder=AdvancedObs(),
                 terminal_conditions=[BallTouchedCondition(), TimeoutCondition(60)],
                 state_setter=CustomStateSetter(),
                 game_speed=1, use_injector=True)

model = PPO("MlpPolicy", env)

def print_obs(cur, last):
    print("____Cur Obs____")
    print(cur)
    # print("___Prev Obs____")
    # print(last)
    # print("_______________")


for _ in range(int(5)):
    obs = env.reset()
    print_obs(obs, model._last_obs)
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    # obs, _, _, _ = env.step([0,0,0,0,0,0,0,0])
    obs, _, _, _ = env.step(action)
    

env.close()
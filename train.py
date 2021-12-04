import rlgym
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from customObsBuilder import BaseObsBuilder
from customRewardFunction import BaseRewardFunction
# from customTerminalCondition import BaseTerminalCondition
from customStateSetter import CustomStateSetter
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.obs_builders import AdvancedObs
import pickle
from wandb.integration.sb3 import WandbCallback
import wandb

def get_env():
    reward_fn = BaseRewardFunction()
    obs_builder = AdvancedObs()
    # terminal_condition = BaseTerminalCondition()
    timeout_condition = TimeoutCondition(70) #Multiply by .0666 to get # seconds. 150timesteps = 10 seconds
    state_setter = CustomStateSetter()
    # terminal_condition = BaseTerminalCondition()
    env = rlgym.make(
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        terminal_conditions=[timeout_condition, GoalScoredCondition()],
        # state_setter=state_setter,
        game_speed=100
    )
    env = Monitor(env)
    return env


def get_wandb_run():
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": int(5e6),
    "env_name": "rlgym"
    }
    run = wandb.init(
        project="rlgym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    return run


def train(isAnUpdate, update, new):
    config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "rlgym",
    }
    wandb_run = get_wandb_run()
    env = get_env()
    model = None
    if isAnUpdate:
        model = PPO.load("models/" + update, env=env, tensorboard_log=f"runs/{wandb_run.id}")
    else:
        model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log=f"runs/{wandb_run.id}")

    try:
        model.learn(total_timesteps=int(5e3),
            callback=WandbCallback(
                verbose=2, gradient_save_freq=100))

        model.save("models/" + new)
        print("Saved Model")

    except KeyboardInterrupt:
        model.save("models/" + new)
        print("Sucessfully Interrupted: Saved Model")
    finally:
        env.close()

def run(to_run):
    print('RUNNINNGGG')
    env = get_env()
    model = PPO.load("models/" + to_run, env=env)
    # model = PPO("MlpPolicy", env=env, verbose=1)
    for _ in range(int(1e4)):
        obs = env.reset()
        for i in range(60): #timesteps per episode
            if i == 0:
                print("___Last Obs____")
                print(model._last_obs)
                print("____Curren Obs__")
                print(obs)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
    
    # model.save(f"models/{to_run}chkpt")
    # print('saved model checkpoint')

if __name__ == "__main__":
    update = ''
    new = ''
    to_run = ''
    isAnUpdate = False
    if sys.argv[1] == 'new':
        new = sys.argv[2]
        print(f'Saving new model to models/{new}')
    elif sys.argv[1] == 'update':
        isAnUpdate = True
        update = sys.argv[2]
        new = sys.argv[3]
        print(f'Loading model {update}')
        print(f'Saving new model to models/{new}')
    elif sys.argv[1] == 'run':
        to_run = sys.argv[2]
        run(to_run)
        quit()
    else:
        print("""Error: invalid inputs.
                Usage: 
                train.py new <save_model_name>
                train.py update <load_model_name> <save_model_name>
                """)
        quit()
    train(isAnUpdate, update, new)

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

class BaseRewardFunction(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = LiuDistancePlayerToBallReward()
        rewardVal = reward.get_reward(player, state, previous_action)
        return rewardVal
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward

from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, TouchBallReward
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from  rlgym.utils.reward_functions.combined_reward import CombinedReward


from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np

def combReward(): 
    reward = CombinedReward.from_zipped(
        (LiuDistancePlayerToBallReward(), .2),
        (VelocityPlayerToBallReward(), 0.1),
        (TouchBallReward(), 0.5),
        (VelocityReward(), 0.2),
    )
    return reward

class BaseRewardFunction(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_combined_reward():
        reward = CombinedReward.from_zipped(
            (LiuDistancePlayerToBallReward(), 1),
            (VelocityPlayerToBallReward(), 0.1),
            (TouchBallReward(), 10000),
            # (VelocityReward(), 0.2),
        )
        return reward

    def get_distance(self, player, state):
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS


    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        #reward1 = LiuDistancePlayerToBallReward().get_reward(player, state, previous_action)
        # reward2 = LiuDistanceBallToGoalReward().get_reward(player, state, previous_action)
       # reward3 = 1000*player.ball_touched
        #rewardVal = reward1+reward3
        '''reward = CombinedReward.from_zipped(
            (EventReward(goal=1, concede=-1), .5),
            (VelocityPlayerToBallReward(), 0.05),
            (VelocityBallToGoalReward(), 0.2),
            (TouchBallReward(), 0.08),
            (VelocityReward(), 0.11),
        )
        return reward
        '''
        # reward = get_combined_reward().get_reward(player, state, previous_action)
        
        reward = -10 * self.get_distance(player, state) + 1000*TouchBallReward().get_reward(player, state, previous_action)
        # reward = TouchBallReward().get_reward(player, state, previous_action)
        
        return reward


    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
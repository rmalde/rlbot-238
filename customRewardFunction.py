from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import LiuDistanceBallToGoalReward

from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, TouchBallReward
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward, EventReward
from rlgym.utils.reward_functions.common_rewards import VelocityReward
from  rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y, BALL_RADIUS


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
        self.score = initial_state.blue_score

    def get_combined_reward():
        reward = CombinedReward.from_zipped(
            (LiuDistancePlayerToBallReward(), 1),
            (VelocityPlayerToBallReward(), 0.1),
            (TouchBallReward(), 10000),
            # (VelocityReward(), 0.2),
        )
        return reward

    def get_distance_player_to_ball(self, player, state):
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        return dist

    def get_distance_ball_to_goal(self, player, state):
        objective = None
        if player.team_num == BLUE_TEAM:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return dist

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
        
        # reward = -0.0001 * self.get_distance_player_to_ball(player, state) + TouchBallReward().get_reward(player, state, previous_action) * 1000
        # reward = TouchBallReward().get_reward(player, state, previous_action)
        # reward = EventReward(goal=1000.).get_reward(player, state, previous_action)
        reward = -0.0001 * self.get_distance_ball_to_goal(player, state) - \
                0.00001 * self.get_distance_player_to_ball(player, state)
        return reward


    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        scored = False
        if state.blue_score > self.score:
            scored = True
            # self.score += 1
            print("Scored!!")
            print(f'Current score: {self.score}')
            print(f'Blue score: {state.blue_score}')
        return 1000*scored
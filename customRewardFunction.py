from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.player_ball_rewards import LiuDistancePlayerToBallReward

class BaseRewardFunction(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = LiuDistancePlayerToBallReward()

        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0
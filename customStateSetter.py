from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np

class CustomStateSetter(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
    spawn_pos = 1

    # def __init__(self, spawn_pos):
    #     self.spawn_pos = spawn_pos

    def reset(self, state_wrapper: StateWrapper):
        # Loop over every car in the game.
        pos, yaw = None, None
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                pos = self.SPAWN_BLUE_POS[self.spawn_pos]
                yaw = self.SPAWN_BLUE_YAW[self.spawn_pos]

            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = self.SPAWN_ORANGE_POS[self.spawn_pos]
                yaw = self.SPAWN_ORANGE_YAW[self.spawn_pos]

            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            if pos or yaw is None:
                print("pos or yaw is None which it shouldn't be")
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
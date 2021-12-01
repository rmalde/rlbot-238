from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState


class BaseTerminalCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        #ends when car touches the ball
        if current_state.last_touch != -1:
            print("touched ball, terminated")
            return True
        else:
            return False
        # return False
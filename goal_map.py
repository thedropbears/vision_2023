from helper_types import GoalState, GoalRegionObservation
from typing import List


class GoalMap:
    def __init__(self):
        pass

    def update(self, goal_observations: List[GoalRegionObservation]):
        pass

    def get_state(self) -> List[GoalState]:
        pass

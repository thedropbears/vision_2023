from helper_types import GoalState, GoalRegionObservation


class GoalMap:
    def __init__(self):
        pass

    def update(self, goal_observations: list[GoalRegionObservation]):
        pass

    def get_state(self) -> list[GoalState]:
        pass

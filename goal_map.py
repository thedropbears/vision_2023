from helper_types import (
    GoalRegionState,
    GoalRegionObservation,
    GoalRegion,
    ExpectedGamePiece,
)
from typing import List
from wpimath.geometry import Translation3d


class GoalRegionMap:
    X_OFFSET_TO_GRID: float = 0.0
    Y_OFFSET_TO_GRID: float = 516.763
    INTAKE_ZONE_OFFSET: float = 0.0
    Y_DISTANCE_BETWEEN_GOALS: float = 558.0

    Z_DISTANCE_CUBE_LOOK_UP: List[float] = [1169.988, 865.188, 0.0]
    Z_DISTANCE_CONE_LOOK_UP: List[float] = [
        826.326 + 120.0,
        522.288 + 120.0,
        0.0 + 120.0,
    ]

    X_DISTANCE_CUBE_LOOK_UP: List[float] = [353.22, 796.777, 1197.227]
    X_DISTANCE_CONE_LOOK_UP: List[float] = [364.231, 795.231, 1167.655]

    def __init__(self, on_blue_alliance: bool):
        self.map: List[GoalRegionState] = []

        # This will start from the top row of goals
        for row in range(3):
            cone_height = self.Z_DISTANCE_CONE_LOOK_UP[row]
            cube_height = self.Z_DISTANCE_CUBE_LOOK_UP[row]
            # this will start from the grid closest to the field origin
            for grid in range(3):
                # this will start from the goal closest to the field origin in each grid
                for goal in range(3):
                    id = row * 9 + grid * 3 + goal
                    expected_game_piece = ExpectedGamePiece.CONE
                    position = Translation3d()
                    position.y = self.Y_DISTANCE_BETWEEN_GOALS * (grid + goal)
                    if not on_blue_alliance:
                        position.y += self.INTAKE_ZONE_OFFSET
                    position.z = cone_height
                    position.x = self.X_DISTANCE_CONE_LOOK_UP[grid]

                    if row == 2:
                        expected_game_piece = ExpectedGamePiece.BOTH
                        position.z = cube_height
                        position.x = self.X_DISTANCE_CUBE_LOOK_UP[grid]
                    elif goal == 1:
                        expected_game_piece = ExpectedGamePiece.CUBE
                        position.z = cube_height
                        position.x = self.X_DISTANCE_CUBE_LOOK_UP[grid]

                    self.map.append[
                        (GoalRegionState(GoalRegion(id, expected_game_piece, position)))
                    ]

    def update(self, goal_observations: list[GoalRegionObservation]):
        pass

    def get_state(self) -> List[GoalRegionState]:
        return self.map

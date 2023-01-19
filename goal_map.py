from helper_types import (
    NodeRegionState,
    NodeRegionObservation,
    NodeRegion,
    ExpectedGamePiece,
)
from wpimath.geometry import Translation3d


class NodeRegionMap:
    X_OFFSET_TO_GRID: float = 0.0
    Y_OFFSET_TO_GRID: float = 516.763
    INTAKE_ZONE_OFFSET: float = 0.0
    Y_DISTANCE_BETWEEN_NODES: float = 558.0

    Z_DISTANCE_CUBE_LOOK_UP: list[float] = [1169.988, 865.188, 0.0]
    Z_DISTANCE_CONE_LOOK_UP: list[float] = [
        826.326 + 120.0,
        522.288 + 120.0,
        0.0 + 120.0,
    ]

    X_DISTANCE_CUBE_LOOK_UP: list[float] = [353.22, 796.777, 1197.227]
    X_DISTANCE_CONE_LOOK_UP: list[float] = [364.231, 795.231, 1167.655]

    def __init__(self, on_blue_alliance: bool):
        self.map: list[NodeRegionState] = []

        # This will start from the top row of nodes
        for row in range(3):
            cone_height = self.Z_DISTANCE_CONE_LOOK_UP[row]
            cube_height = self.Z_DISTANCE_CUBE_LOOK_UP[row]
            # this will start from the grid closest to the field origin
            for grid in range(3):
                # this will start from the node closest to the field origin in each grid
                for node in range(3):
                    id = row * 9 + grid * 3 + node
                    expected_game_piece = ExpectedGamePiece.CONE
                    pos_x = 0.0
                    pos_y = 0.0
                    pos_z = 0.0

                    pos_y = self.Y_DISTANCE_BETWEEN_NODES * (grid + node)
                    if not on_blue_alliance:
                        pos_y += self.INTAKE_ZONE_OFFSET
                    pos_z = cone_height
                    pos_x = self.X_DISTANCE_CONE_LOOK_UP[grid]

                    if row == 2:
                        expected_game_piece = ExpectedGamePiece.BOTH
                        pos_z = cube_height
                        pos_x = self.X_DISTANCE_CUBE_LOOK_UP[grid]
                    elif node == 1:
                        expected_game_piece = ExpectedGamePiece.CUBE
                        pos_z = cube_height
                        pos_x = self.X_DISTANCE_CUBE_LOOK_UP[grid]

                    position = Translation3d(pos_x, pos_y, pos_z)

                    self.map.append[
                        (NodeRegionState(NodeRegion(id, expected_game_piece, position)))
                    ]

    def update(self, node_observations: list[NodeRegionObservation]):
        pass

    def get_state(self) -> list[NodeRegionState]:
        return self.map

from helper_types import (
    NodeRegionState,
    NodeRegionObservation,
    NodeRegion,
    ExpectedGamePiece,
)
from wpimath.geometry import Translation3d


class NodeRegionMap:
    X_OFFSET_TO_GRID: float = 0.0
    Y_OFFSET_TO_GRID: float = 0.516763
    INTAKE_ZONE_OFFSET: float = 0.0
    Y_DISTANCE_BETWEEN_NODES: float = 0.558

    Z_DISTANCE_CUBE_LOOK_UP: list[float] = [1.169988, 0.865188, 0.0]
    Z_DISTANCE_CONE_LOOK_UP: list[float] = [
        0.826326 + 0.1200,
        0.522288 + 0.1200,
        0.0 + 0.1200,
    ]

    X_DISTANCE_CUBE_LOOK_UP: list[float] = [0.35322, 0.796777, 1.197227]
    X_DISTANCE_CONE_LOOK_UP: list[float] = [0.364231, 0.795231, 1.167655]

    def __init__(self, on_blue_alliance: bool):
        self.node_map: list[NodeRegionState] = []

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

                    self.node_map.append(
                        NodeRegionState(NodeRegion(id, expected_game_piece, position))
                    )

    def update(self, node_observations: list[NodeRegionObservation]):
        pass

    def get_state(self) -> list[NodeRegionState]:
        return self.node_map

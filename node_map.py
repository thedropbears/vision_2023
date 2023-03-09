from enum import Enum
from helper_types import (
    Node,
    GamePiece,
)
from wpimath.geometry import Translation3d

FIELD_LENGTH = 16.540988


class Rows(Enum):
    HIGH = 0
    MID = 1
    LOW = 2


Y_OFFSET_TO_GRID: float = 0.512
Y_DISTANCE_BETWEEN_NODES: float = 0.5588

CUBE_Z_OFFSET = 0.1
CONE_Z_OFFSET = -0.1
SCORING_Z_LOOK_UP = {
    # heights to cube node base
    GamePiece.CUBE: {Rows.HIGH: 0.826 + CUBE_Z_OFFSET, Rows.MID: 0.522 + CUBE_Z_OFFSET},
    # heights to top of pole
    GamePiece.CONE: {Rows.HIGH: 1.170 + CONE_Z_OFFSET, Rows.MID: 0.865 + CONE_Z_OFFSET},
    GamePiece.BOTH: {Rows.LOW: 0.0 + CUBE_Z_OFFSET},
}
SCORING_X_LOOK_UP = {Rows.HIGH: 0.36, Rows.MID: 0.79, Rows.LOW: 1.18}

ALL_NODES: list[Node] = []
for row in Rows:
    for col in range(9):
        if row is Rows.LOW:
            piece = GamePiece.BOTH
        elif col % 3 == 1:
            piece = GamePiece.CUBE
        else:
            piece = GamePiece.CONE

        y = Y_OFFSET_TO_GRID + Y_DISTANCE_BETWEEN_NODES * col
        blue_node_pos = Translation3d(
            SCORING_X_LOOK_UP[row], y, SCORING_Z_LOOK_UP[piece][row]
        )
        red_node_pos = Translation3d(
            FIELD_LENGTH - SCORING_X_LOOK_UP[row], y, SCORING_Z_LOOK_UP[piece][row]
        )
        red_id = row.value * 9 + col + 27
        blue_id = row.value * 9 + col
        ALL_NODES.append(Node(red_id, piece, red_node_pos))
        ALL_NODES.append(Node(blue_id, piece, blue_node_pos))
ALL_NODES = sorted(ALL_NODES, key=lambda n: n.id)

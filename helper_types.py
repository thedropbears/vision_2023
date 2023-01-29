from dataclasses import dataclass
from enum import Enum
from wpimath.geometry import Translation3d


class ExpectedGamePiece(Enum):
    CONE = 1
    CUBE = 2
    BOTH = 3


class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def area(self) -> int:
        return abs((self.x2 - self.x1) * (self.y2 - self.y1))


@dataclass
class NodeRegion:
    id: int
    expected_game_piece: ExpectedGamePiece
    position: Translation3d


@dataclass
class NodeRegionObservation:
    camera_id: int
    bounding_box: BoundingBox
    node_region: NodeRegion


@dataclass
class NodeRegionState:
    node_region: NodeRegion
    occupied: bool = False

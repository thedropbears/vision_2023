from dataclasses import dataclass
from enum import Enum
from wpimath.geometry import Translation3d


class GamePiece(Enum):
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
class Node:
    id: int
    expected_game_piece: GamePiece
    position: Translation3d


@dataclass
class NodeView:
    """A node in a camera"""
    bounding_box: BoundingBox
    node: Node


@dataclass
class NodeObservation:
    """An node view and its state"""
    view: NodeView
    occupied: bool = False

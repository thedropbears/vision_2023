from dataclasses import dataclass
from enum import Enum


class ExpectedGamePiece(Enum):
    CONE = 1
    CUBE = 2
    BOTH = 3


class BoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    def area(self)->int:
        return abs((self.x2-self.x1)*(self.y2-self.y1))



@dataclass
class GoalRegionObservation:
    bounding_box: BoundingBox
    expected_game_piece: ExpectedGamePiece
    expected_id: int


@dataclass
class GoalState:
    actual_id: int
    allowable_game_pieces: ExpectedGamePiece
    occupied: bool

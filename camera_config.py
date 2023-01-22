from dataclasses import dataclass
import numpy as np
from wpimath.geometry import Transform3d
from math import atan2


@dataclass
class CameraParams:
    name: str  # the name of the camera
    width: int
    height: int
    transform: Transform3d  # pose of the camera w.r.t to the base link of the robot
    K: np.ndarray  # [3 x 3] intrinsic camera matrix
    fps: int

    def get_horizontal_fov(self) -> float:
        return 2 * atan2(self.width / 2, self.K[0, 0])

    def get_vertical_fov(self) -> float:
        return 2 * atan2(self.height / 2, self.K[1, 1])

from dataclasses import dataclass
import numpy as np
from wpimath.geometry import Pose3d


@dataclass
class CameraParams:
    name: str # the name of the camera 
    width: int
    height: int
    transform: Pose3d # pose of the camera w.r.t to the base link of the robot
    K: np.ndarray # [3 x 3] intrinsic camera matrix
    fps: int 

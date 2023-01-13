import numpy as np
from math import pi
from wpimath.geometry import Translation3d, Rotation3d, Transform3d

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
CONE_HSV_LOW = np.array([(28 / 240) * 180, (118 / 240) * 255, (107 / 240) * 255])
CONE_HSV_HIGH = np.array([(35 / 240) * 180, (240 / 240) * 255, (220 / 240) * 255])

CUBE_HSV_LOW = np.array([(160 / 240) * 180, (99 / 240) * 255, (59 / 240) * 255])
CUBE_HSV_HIGH = np.array([(185 / 240) * 180, (240 / 240) * 255, (225 / 240) * 255])

CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD = 0.1

# Gate angle to determine if robot can even see the goal or not to avoid projecting backwards
DIRECTION_GATE_ANGLE = pi

ROBOT_BASE_TO_CAMERA_TRANSLATION = Translation3d(0.0, 0.0, 0.0)
ROBOT_BASE_TO_CAMERA_ROTATION = Rotation3d(0.0, 0.0, 0.0)
ROBOT_BASE_TO_CAMERA_TRANSFORMATION = Transform3d(
    ROBOT_BASE_TO_CAMERA_TRANSLATION, ROBOT_BASE_TO_CAMERA_ROTATION
)

CAMERA_MATRIX = np.array((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

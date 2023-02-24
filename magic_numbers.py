import math
import numpy as np
from math import pi
from wpimath.geometry import Translation3d, Rotation3d, Transform3d, Pose3d

from camera_config import CameraParams

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
CONE_HSV_LOW = np.array([(28 / 240) * 180, (118 / 240) * 255, (107 / 240) * 255])
CONE_HSV_HIGH = np.array([(35 / 240) * 180, (240 / 240) * 255, (220 / 240) * 255])

CUBE_HSV_LOW = np.array([(160 / 240) * 180, (99 / 240) * 255, (59 / 240) * 255])
CUBE_HSV_HIGH = np.array([(185 / 240) * 180, (240 / 240) * 255, (225 / 240) * 255])

CONE_HEIGHT = 0.33
CUBE_HEIGHT = 0.24

CONE_WIDTH = 0.21
CUBE_WIDTH = CUBE_HEIGHT

CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD = 0.1

# Gate angle to determine if robot can even see the node or not to avoid projecting backwards
DIRECTION_GATE_ANGLE = pi

ROBOT_BASE_TO_CAMERA_TRANSLATION = Translation3d(0.0, 0.0, 0.0)
ROBOT_BASE_TO_CAMERA_ROTATION = Rotation3d(0.0, 0.0, 0.0)
ROBOT_BASE_TO_CAMERA_TRANSFORMATION = Transform3d(
    ROBOT_BASE_TO_CAMERA_TRANSLATION, ROBOT_BASE_TO_CAMERA_ROTATION
)

CAMERA_MATRIX = np.array(
    [
        [
            950.0960104757881,
            0.0,
            629.0702597777629
        ],
        [
            0.0,
            949.2742671058766,
            348.4667207420139
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])

camera_rotation = Rotation3d(0, 0, math.pi)
camera_translation = Translation3d(-0.5, 0, 0.3)
camera_params1 = CameraParams(
    "Camera",
    FRAME_WIDTH,
    FRAME_HEIGHT,
    Transform3d(Pose3d(), Pose3d(camera_translation, camera_rotation)),
    CAMERA_MATRIX,
    30,
)
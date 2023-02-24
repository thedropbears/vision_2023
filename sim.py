from camera_manager import WebcamCameraManager
from connection import NTConnection
from magic_numbers import camera_params1
from vision import GamePieceVision
from wpimath.geometry import Pose2d

left_camera = WebcamCameraManager(0, camera_params1)

vision = GamePieceVision(
    left_camera,
    NTConnection("left_cam"),
)
while True:
    vision.run()

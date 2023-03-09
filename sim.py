from camera_manager import WebcamCameraManager
from connection import NTConnection
from magic_numbers import camera_params1
from vision import GamePieceVision as Vision

camera = WebcamCameraManager(0, camera_params1)

connection = NTConnection("left_cam", sim=True)

vision = Vision(
    camera,
    connection,
)

while True:
    vision.run()
    # connection.pose = connection.pose.transformBy(Transform2d(0.0, 0.0, 0.05))

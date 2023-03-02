from camera_manager import WebcamCameraManager
from connection import DummyConnection, NTConnection
from magic_numbers import camera_params1
from vision import GamePieceVision as Vision
from wpimath.geometry import Transform2d

left_camera = WebcamCameraManager(0, camera_params1)
connection = NTConnection(
    "vision"
)

vision = Vision(
    left_camera,
    connection,
)

while True:
    vision.run()
    # connection.pose = connection.pose.transformBy(Transform2d(0.0, 0.0, 0.05))

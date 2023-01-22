from camera_manager import CameraManager
from connection import NTConnection
from magic_numbers import (
    CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD,
    CONE_HSV_HIGH,
    CONE_HSV_LOW,
    CUBE_HSV_HIGH,
    CUBE_HSV_LOW,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

from math import atan2
import cv2
import numpy as np
from helper_types import (
    NodeRegionObservation,
    NodeRegion,
    NodeRegionState,
    ExpectedGamePiece,
    BoundingBox,
)
from camera_config import CameraParams
from node_map import NodeRegionMap
from wpimath.geometry import Pose2d, Pose3d, Translation3d, Transform3d, Rotation3d


class Vision:
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection
        self.map = NodeRegionMap()

    def run(self) -> None:
        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """
        self.connection.pong()
        frame_time, frame = self.camera_manager.get_frame()
        # frame time is 0 in case of an error
        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return

        results, display = self.process_image(frame)

        if results is not None:
            # TODO send results to rio
            pass

        # send image to display on driverstation
        self.camera_manager.send_frame(display)
        self.connection.set_fps()


def is_coloured_game_piece(
    masked_image: np.ndarray,
    lower_colour: np.ndarray,
    upper_colour: np.ndarray,
    bBox_area: int,
) -> bool:

    gamepiece_mask = cv2.inRange(masked_image, lower_colour, upper_colour)

    # get largest contour
    contours, hierarchy = cv2.findContours(
        gamepiece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) > 0:
        # find largest contour in mask, use to compute minEnCircle
        biggest_contour = max(contours, key=cv2.contourArea)
        # get area of contour
        area = cv2.contourArea(biggest_contour)
        return area / bBox_area > CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD

    else:
        return False


def is_game_piece_present(
    frame: np.ndarray, bounding_box: BoundingBox, expected_game_piece: ExpectedGamePiece
) -> bool:

    # draw bound box mask
    bBox_mask = np.zeros_like(frame)
    bBox_mask = cv2.rectangle(
        bBox_mask,
        (bounding_box.x1, bounding_box.y1),
        (bounding_box.x2, bounding_box.y2),
        (255, 255, 255),
        cv2.FILLED,
    )

    masked_image = cv2.bitwise_and(frame, bBox_mask)

    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    cube_present = False
    cone_present = False
    if (
        expected_game_piece == ExpectedGamePiece.BOTH
        or expected_game_piece == ExpectedGamePiece.CUBE
    ):
        # run cube mask
        lower_purple = CUBE_HSV_LOW
        upper_purple = CUBE_HSV_HIGH
        cube_present = is_coloured_game_piece(
            hsv, lower_purple, upper_purple, bounding_box.area()
        )

    if (
        expected_game_piece == ExpectedGamePiece.BOTH
        or expected_game_piece == ExpectedGamePiece.CONE
    ):
        # run cone mask
        lower_yellow = CONE_HSV_LOW
        upper_yellow = CONE_HSV_HIGH

        cone_present = is_coloured_game_piece(
            hsv, lower_yellow, upper_yellow, bounding_box.area()
        )

    return cone_present or cube_present


def process_image(frame: np.ndarray, pose: Pose2d):

    # visible_nodes = self.find_visible_nodes(frame, pose)

    # node_states = self.detect_node_state(frame, visible_nodes)

    # whatever the update step is
    # self.map.update(node_states)

    # map_state = self.map.get_state()

    # annotate frame
    # annotated_frame = annotate_image(frame, map_state, node_states)

    # return state of map (state of all node regions) and annotated camera stream
    return


def find_visible_nodes(frame: np.ndarray, pose: Pose2d) -> list[NodeRegionObservation]:
    """Segment image to find visible nodes in a frame

    Args:
        frame (np.ndarray): New camera frame containing nodes
        pose (Pose2d): Current robot pose in the world frame

    Returns:
        List[NodeRegionObservation]: List of node region observations with no information about occupancy
    """
    pass


def detect_node_state(
    frame: np.ndarray, regions_of_interest: list[NodeRegionObservation]
) -> list[NodeRegionObservation]:
    """Detect node occupancy in a set of observed node regions

    Args:
        frame (np.ndarray): New camera frame containing nodes
        regions_of_interest (List[NodeRegionObservation]): List of node region observations with no information about occupancy

    Returns:
        List[NodeRegionObservation]: List of node region observations
    """
    pass


def annotate_image(
    frame: np.ndarray,
    map: list[NodeRegionState],
    node_observations: list[NodeRegionObservation],
) -> np.ndarray:
    """annotate a frame with projected node points

    Args:
        frame (np.ndarray): raw image frame without annotation
        map (List[NodeState]): current map state with information on occupancy state_
        node_observations (List[NodeRegionObservation]): node observations in the current time step

    Returns:
        np.ndarray: frame annotated with observed node regions
    """
    pass


def is_node_region_in_image(
    robot_pose: Pose2d,
    camera_params: CameraParams,
    node_region: NodeRegion,
) -> bool:

    # create transform to make camera origin
    world_to_robot = Transform3d(Pose3d(), Pose3d(robot_pose))
    world_to_camera = world_to_robot + camera_params.transform
    node_region_camera_frame = world_to_camera.inverse() + Transform3d(
        node_region.position, Rotation3d()
    )

    # Check the robot is facing the right direction for the point by checking it is inside the FOV
    return point3d_in_field_of_view(
        node_region_camera_frame.translation(), camera_params
    )


def point3d_in_field_of_view(point: Translation3d, camera_params: CameraParams) -> bool:
    """Determines if a point in 3d space relative to the camera coordinate frame is visible in a camera's field of view

    Args:
        point (Translation3d): _point in 3d space relative to a camera
        camera_params (CameraParams): camera parameters structure providing information about a frame being processed

    Returns:
        bool: if point is visible
    """
    vertical_angle = atan2(point.z, point.x)
    horizontal_angle = atan2(point.y, point.x)

    return (
        (point.x > 0)
        and (
            -camera_params.get_vertical_fov() / 2
            < vertical_angle
            < camera_params.get_vertical_fov() / 2
        )
        and (
            -camera_params.get_horizontal_fov() / 2
            < horizontal_angle
            < camera_params.get_horizontal_fov() / 2
        )
    )


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # To update

    params = CameraParams("Camera", FRAME_WIDTH, FRAME_HEIGHT, Translation3d(), K, 30)

    vision = Vision(
        CameraManager(
            "/dev/video0",
            params,
            "kYUYV",
        ),
        NTConnection(),
    )
    while True:
        vision.run()

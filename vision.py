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
    GoalRegionObservation,
    GoalRegion,
    GoalRegionState,
    ExpectedGamePiece,
    BoundingBox,
)
from camera_config import CameraParams
from goal_map import GoalRegionMap
from wpimath.geometry import Pose2d, Pose3d, Translation3d, Transform3d, Rotation3d


class Vision:
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection
        self.map = GoalRegionMap()

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

    # visible_goals = self.find_visible_goals(frame, pose)

    # goal_states = self.detect_goal_state(frame, visible_goals)

    # whatever the update step is
    # self.map.update(goal_states)

    # map_state = self.map.get_state()

    # annotate frame
    # annotated_frame = annotate_image(frame, map_state, goal_states)

    # return state of map (state of all goal regions) and annotated camera stream
    return


def find_visible_goals(frame: np.ndarray, pose: Pose2d) -> list[GoalRegionObservation]:
    """Segment image to find visible goals in a frame

    Args:
        frame (np.ndarray): New camera frame containing goals
        pose (Pose2d): Current robot pose in the world frame

    Returns:
        List[GoalRegionObservation]: List of goal region observations with no information about occupancy
    """
    pass


def detect_goal_state(
    frame: np.ndarray, regions_of_interest: list[GoalRegionObservation]
) -> list[GoalRegionObservation]:
    """Detect goal occupancy in a set of observed goal regions

    Args:
        frame (np.ndarray): New camera frame containing goals
        regions_of_interest (List[GoalRegionObservation]): List of goal region observations with no information about occupancy

    Returns:
        List[GoalRegionObservation]: List of goal region observations
    """
    pass


def annotate_image(
    frame: np.ndarray,
    map: list[GoalRegionState],
    goal_observations: list[GoalRegionObservation],
) -> np.ndarray:
    """annotate a frame with projected goal points

    Args:
        frame (np.ndarray): raw image frame without annotation
        map (List[GoalState]): current map state with information on occupancy state_
        goal_observations (List[GoalRegionObservation]): goal observations in the current time step

    Returns:
        np.ndarray: frame annotated with observed goal regions
    """
    pass


def is_goal_region_in_image(
    frame: np.ndarray,
    robot_pose: Pose2d,
    camera_params: CameraParams,
    goal_region: GoalRegion,
) -> bool:

    # create transform to make camera origin
    world_to_robot = Transform3d(Pose3d(), Pose3d(robot_pose))
    world_to_camera = world_to_robot + camera_params.transform
    goal_region_camera_frame = world_to_camera.inverse() + Transform3d(
        goal_region.position, Rotation3d()
    )

    # Check the robot is facing the right direction for the point by checking it is inside the FOV
    if not point3d_in_field_of_view(
        goal_region_camera_frame.translation(), camera_params
    ):
        return False

    # extract list of goal points for use in opencv api
    goal_point_image_coordinates = []

    # Project point into pixel space
    cv2.projectPoints(
        objectPoints=np.array(list[goal_region_camera_frame.translation()]),
        camera_matrix=camera_params.K,
        imagePoints=goal_point_image_coordinates,
    )

    # check image boundaries to determine which goals are actually in the image
    return point2d_in_image_frame(goal_point_image_coordinates[0], frame)


def point3d_in_field_of_view(point: Translation3d, camera_params: CameraParams) -> bool:
    vertical_angle = atan2(point.y(), point.z())
    horizontal_angle = atan2(point.x(), point.z())
    return (
        (point.z() < 0)
        and not (
            vertical_angle > -camera_params.get_vertical_fov() / 2
            and vertical_angle < camera_params.get_vertical_fov() / 2
        )
        and not (
            horizontal_angle > -camera_params.get_horizontal_fov()
            and horizontal_angle < camera_params.get_horizontal_fov() / 2
        )
    )


def point2d_in_image_frame(pixel: np.ndarray, frame: np.ndarray):
    return (
        (pixel[0] >= 0)
        and (pixel[0] < frame.shape[1])
        and (pixel[1] >= 0)
        and (pixel[1] < frame.shape[0])
    )


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # To update

    params = CameraParams("Camera", FRAME_WIDTH, FRAME_HEIGHT, Pose3d(), K, 30)

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

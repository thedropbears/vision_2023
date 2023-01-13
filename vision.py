from camera_manager import CameraManager, CameraParams
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
import cv2
import numpy as np
from helper_types import (
    GoalRegionObservation,
    GoalState,
    ExpectedGamePiece,
    BoundingBox,
)
from goal_map import GoalMap
from wpimath.geometry import Pose2d, Pose3d, Translation3d, Transform3d
import sys


class Vision:
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection
        self.map = GoalMap()

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
    map: list[GoalState],
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


def is_goal_in_image(frame: np.ndarray, robot_pose: Pose2d, goal_point: Transform3d 3d):

    # Check the robot is facing the right direction for the point
    if not robot_is_facing_goal(robot_pose):
        return False

    # transform to make camera origin
    # TODO Finish this section
    # goal_point = is in world frame
    # robot_pose = world to_robot
    # robot_to_cam = robot_to cam
    # point_camera_frame = inv(robot_pose * robot_to_cam)) * point

    # Project point into pixel space
    x_p,y_p = project_point_to_image_frame(point_camera_frame, CAMERA_MATRIX)

    u = x_p + len(frame[0])/2
    v = -y_p + len(frame[1])/2

    # check pixel is within the bounds of the frame
    return (u > 0) and (u < FRAME_WIDTH) and (v > 0) and (v < FRAME_HEIGHT)


def robot_is_facing_goal(robot_pose: Pose2d) -> bool:
    return (
        robot_pose.rotation() > DIRECTION_GATE_ANGLE
        or robot_pose.rotation() < -DIRECTION_GATE_ANGLE
    )


def project_point_to_image_frame(point: Translation3d, camera_matrix: np.ndarray) -> Tuple(int, int):
    u = (point.x * camera_matrix[0][0])/point.z + camera_matrix[0][2]
    v = (point.y * camera_matrix[1][1]/point.z) + camera_matrix[1][2]
    return u,v


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

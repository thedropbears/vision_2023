from camera_manager import CameraManager, CameraParams
from connection import NTConnection
from magic_numbers import *
from typing import Tuple, Optional, List
from math import tan
import cv2
import numpy as np
import time
from helper_types import (
    GoalRegionObservation,
    GoalState,
    ExpectedGamePiece,
    BoundingBox,
)
from goal_map import GoalMap
from wpimath.geometry import Pose2d, Pose3d
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
        if area / bBox_area > CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD:
            return True
        else:
            return False

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


def find_visible_goals(frame: np.ndarray, pose: Pose2d) -> List[GoalRegionObservation]:
    """Segment image to find visible goals in a frame

    Args:
        frame (np.ndarray): New camera frame containing goals
        pose (Pose2d): Current robot pose in the world frame

    Returns:
        List[GoalRegionObservation]: List of goal region observations with no information about occupancy
    """
    pass


def detect_goal_state(
    frame: np.ndarray, regions_of_interest: List[GoalRegionObservation]
) -> List[GoalRegionObservation]:
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
    map: List[GoalState],
    goal_observations: List[GoalRegionObservation],
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


if __name__ == "__main__":
    # this is to run on the robot
    # to run vision code on your laptop use sim.py
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #To update
    
    params = CameraParams("Camera", FRAME_WIDTH, FRAME_HEIGHT, Pose3d(),  K, 30)

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

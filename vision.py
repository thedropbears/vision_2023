from camera_manager import CameraManager
from connection import NTConnection
import magic_numbers
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
from wpimath.geometry import Pose2d


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


def is_game_piece_present(
    frame: np.ndarray, bounding_box: BoundingBox, expected_game_piece: ExpectedGamePiece
) -> bool:
    return False


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

    vision = Vision(
        CameraManager(
            "Camera",
            "/dev/video0",
            magic_numbers.FRAME_HEIGHT,
            magic_numbers.FRAME_WIDTH,
            30,
            "kYUYV",
        ),
        NTConnection(),
    )
    while True:
        vision.run()

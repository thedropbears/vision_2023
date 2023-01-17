import csv
import pytest
import cv2
import vision
from helper_types import BoundingBox, ExpectedGamePiece
from goal_map import GoalRegionMap
from wpimath.geometry import Pose2d
import numpy as np


def read_test_data_csv(fname: str):
    with open(fname) as f:
        result = []
        for image, cone_present, cube_present, x1, y1, x2, y2 in csv.reader(f):
            result.append(
                (
                    image,
                    cone_present == "True",
                    cube_present == "True",
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            )
    return result


def read_goal_region_in_frame_csv(fname: str):
    with open(fname) as f:
        result = []
        for (
            image,
            goal_region_visible,
            robot_x,
            robot_y,
            heading,
            goal_region_id,
        ) in csv.reader(f):
            result.append(
                (
                    image,
                    goal_region_visible == "True",
                    float(robot_x),
                    float(robot_y),
                    float(heading),
                    int(goal_region_id),
                )
            )
    return result


images = read_test_data_csv("test/expected.csv")
goal_region_in_frame_images = read_goal_region_in_frame_csv(
    "test/goal_region_in_frame.csv"
)


@pytest.mark.parametrize("filename,cone_present,cube_present,x1,y1,x2,y2", images)
def test_sample_images(
    filename: str,
    cone_present: bool,
    cube_present: bool,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
):
    image = cv2.imread(f"./test/{filename}")
    assert image is not None
    bounding_box = BoundingBox(x1, y1, x2, y2)
    if cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            is True
        ), "Cone present in image but not found by detector"

        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            is False
        ), "Cone present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.BOTH)
            is True
        ), "Cone present in image but detector found neither"

    if cube_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            is True
        ), "Cube present in image but not found by detector"

        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            is False
        ), "Cube present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.BOTH)
            is True
        ), "Cube present in image but detector found neither"

    if not cube_present and not cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CUBE)
            is False
        ), "Nothing present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, ExpectedGamePiece.CONE)
            is False
        ), "Nothing present in image but detector found cone"


@pytest.mark.parametrize(
    "filename,goal_region_visible,robot_x,robot_y,heading, goal_region_id",
    goal_region_in_frame_images,
)
def test_goal_region_in_frame(
    filename: str,
    goal_region_visible: bool,
    robot_x: float,
    robot_y: float,
    heading: float,
    goal_region_id: int,
):
    image = cv2.imread(f"./test/{filename}")
    assert image is not None
    goal_region_map = GoalRegionMap()

    goal_region = goal_region_map.get_state()[goal_region_id].goal_region

    robot_pose = Pose2d(robot_x, robot_y, heading)

    camera_matrix = np.array([1, 0, 0], [0, 1, 0], [0, 0, 1])

    assert (
        vision.is_goal_region_in_image(image, robot_pose, camera_matrix, goal_region)
        == goal_region_visible
    )

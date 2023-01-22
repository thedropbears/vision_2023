import csv
import pytest
import cv2
import vision
from helper_types import BoundingBox, ExpectedGamePiece
from node_map import NodeRegionMap
import numpy as np
from camera_config import CameraParams
from wpimath.geometry import Transform3d, Translation3d, Rotation3d, Rotation2d, Pose2d


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


def read_node_region_in_frame_csv(fname: str):
    with open(fname) as f:
        result = []
        for (
            node_region_visible,
            robot_x,
            robot_y,
            heading,
            node_region_id,
        ) in csv.reader(f):
            result.append(
                (
                    node_region_visible == "True",
                    float(robot_x),
                    float(robot_y),
                    float(heading),
                    int(node_region_id),
                )
            )
    return result


images = read_test_data_csv("test/expected.csv")
node_region_in_frame_images = read_node_region_in_frame_csv(
    "test/node_region_in_frame.csv"
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
    "node_region_visible,robot_x,robot_y,heading,node_region_id",
    node_region_in_frame_images,
)
def test_is_node_region_in_image(
    node_region_visible: bool,
    robot_x: float,
    robot_y: float,
    heading: float,
    node_region_id: int,
):
    node_region_map = NodeRegionMap(on_blue_alliance=True)

    node_region = node_region_map.get_state()[node_region_id].node_region

    robot_pose = Pose2d(robot_x, robot_y, Rotation2d.fromDegrees(heading))

    extrinsic_robot_to_camera = Transform3d(
        Translation3d(-0.35, 0.005, 0.26624),
        Rotation3d.fromDegrees(0, 0, 180),
    )
    intrinsic_camera_matrix = np.array(
        [
            [1.12899023e03, 0.00000000e00, 6.34655248e02],
            [0.00000000e00, 1.12747666e03, 3.46570772e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    camera_params = CameraParams(
        "test_name", 1280, 720, extrinsic_robot_to_camera, intrinsic_camera_matrix, 30
    )

    assert (
        vision.is_node_region_in_image(robot_pose, camera_params, node_region)
        == node_region_visible
    )


def test_point_3d_in_field_of_view():
    # create dummy camera matrix
    extrinsic_robot_to_camera = Transform3d(
        Translation3d(0.0, 0.0, 0.0), Rotation3d(0, 0, 0)
    )
    intrinsic_camera_matrix = np.array(
        [
            [1.12899023e03, 0.00000000e00, 6.34655248e02],
            [0.00000000e00, 1.12747666e03, 3.46570772e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    camera_params = CameraParams(
        "test_name", 1280, 720, extrinsic_robot_to_camera, intrinsic_camera_matrix, 30
    )

    # dummy points for in frame and out of frame
    # at 5m the range from the camera perspective is -2.83m to 2.83m horizontally and  1.6m to -1.6m vertically
    point_in_frame = Translation3d(5.0, 0.1, 0.1)
    point_left_of_frame = Translation3d(5.0, 3.0, 0)
    point_right_of_frame = Translation3d(5.0, -3.0, 0.0)
    point_above_frame = Translation3d(5.0, 0.0, 2.0)
    point_below_frame = Translation3d(5.0, 0.0, -2.0)
    point_behind_frame = Translation3d(-5.0, 0.0, 0.0)

    # assert results from function based on points
    assert (
        vision.point3d_in_field_of_view(point_in_frame, camera_params) is True
    ), "point should be in fov"
    assert (
        vision.point3d_in_field_of_view(point_left_of_frame, camera_params) is False
    ), "point should not be in fov"
    assert (
        vision.point3d_in_field_of_view(point_right_of_frame, camera_params) is False
    ), "point should not be in fov"
    assert (
        vision.point3d_in_field_of_view(point_above_frame, camera_params) is False
    ), "point should not be in fov"
    assert (
        vision.point3d_in_field_of_view(point_below_frame, camera_params) is False
    ), "point should not be in fov"
    assert (
        vision.point3d_in_field_of_view(point_behind_frame, camera_params) is False
    ), "point should not be in fov"

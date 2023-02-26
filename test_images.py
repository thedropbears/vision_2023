import csv
import pytest
import cv2
from node_map import ALL_NODES
import vision
from helper_types import (
    BoundingBox,
    GamePiece,
    NodeView,
    Node,
)
from magic_numbers import camera_params1
import numpy as np
from camera_config import CameraParams
from wpimath.geometry import (
    Transform3d,
    Translation3d,
    Rotation3d,
    Rotation2d,
    Pose2d,
    Pose3d,
)
from vision import GamePieceVision
import camera_manager
import connection
import json
import math


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


def read_visible_node_json(fname: str):
    result = []
    with open(fname) as f:
        fjson = json.load(f)

        for image in fjson["img"]:
            visible_nodes = []
            for node in image["nodes"]:
                visible_nodes.append((node["row"] * 9 + node["col"], node["contain"]))
            result.append(
                (
                    image["img_name"],
                    image["location"]["x"],
                    image["location"]["y"],
                    image["location"]["z"],
                    image["location"]["omega"],
                    visible_nodes,
                )
            )

    return result


classify_images = read_test_data_csv("test/expected.csv")


@pytest.mark.parametrize(
    "filename,cone_present,cube_present,x1,y1,x2,y2", classify_images
)
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

    cam_list: camera_manager.MockImageManager = camera_manager.MockImageManager(
        image, camera_params1
    )
    vision = GamePieceVision(
        cam_list, connection.DummyConnection(Pose2d(), False, False)
    )

    assert image is not None
    bounding_box = BoundingBox(x1, y1, x2, y2)
    if cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CONE) is True
        ), "Cone present in image but not found by detector"

        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CUBE) is False
        ), "Cone present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.BOTH) is True
        ), "Cone present in image but detector found neither"

    if cube_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CUBE) is True
        ), "Cube present in image but not found by detector"

        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CONE) is False
        ), "Cube present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.BOTH) is True
        ), "Cube present in image but detector found neither"

    if not cube_present and not cone_present:
        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CUBE) is False
        ), "Nothing present in image but detector found cube"

        assert (
            vision.is_game_piece_present(image, bounding_box, GamePiece.CONE) is False
        ), "Nothing present in image but detector found cone"


node_region_in_frame_images = read_node_region_in_frame_csv(
    "test/node_region_in_frame.csv"
)


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
    matching_nodes = [x for x in ALL_NODES if x.id == node_region_id]
    assert (
        len(matching_nodes) == 1
    ), f"Invalid node id {node_region_id}, matches: {len(matching_nodes)}"
    node = matching_nodes[0]
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

    camera_pose = GamePieceVision.get_camera_pose(robot_pose, camera_params)

    assert (
        vision.is_node_in_image(camera_pose, camera_params, node) == node_region_visible
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
    point_in_frame_left = Translation3d(5.0, 2, 0)
    point_in_frame_above = Translation3d(5.0, 0, 1)
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
        vision.point3d_in_field_of_view(point_in_frame_left, camera_params) is True
    ), "point should be in fov"
    assert (
        vision.point3d_in_field_of_view(point_in_frame_above, camera_params) is True
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


json_nodes = read_visible_node_json("test/expected.json")


@pytest.mark.parametrize(
    "image_name,x,y,z,heading,json_visible_nodes",
    json_nodes,
)
def test_find_visible_nodes(
    image_name: str,
    x: float,
    y: float,
    z: float,
    heading: float,
    json_visible_nodes: tuple,
):
    image = cv2.imread(f"./test/test_images/{image_name}")

    extrinsic_trans = Transform3d()
    intrinsic_camera_matrix = np.array(
        [
            [1.12899023e03, 0.00000000e00, 6.34655248e02],
            [0.00000000e00, 1.12747666e03, 3.46570772e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=np.float32,
    )

    camera_params = CameraParams(
        "test_name", 1280, 720, extrinsic_trans, intrinsic_camera_matrix, 30
    )

    cam = camera_manager.MockImageManager(image, camera_params)
    vision = GamePieceVision(cam, connection.DummyConnection())

    pose = Pose3d(Translation3d(x, y, z), Rotation3d(0, 0, math.radians(heading)))
    observed_nodes = vision.find_visible_nodes(image, pose)
    observed_nodes_ids = [x.node.id for x in observed_nodes]

    should_see_nodes = [n[0] for n in json_visible_nodes]
    print(should_see_nodes, observed_nodes_ids)
    assert all(
        [seen_node in observed_nodes_ids for seen_node in should_see_nodes]
    ), "visible nodes all observed in test_find_visible_nodes"

    # assert {
    #     observed_nodes_id != json_visible_nodes[0]
    # }, "visible nodes not fully observed in test_find_visible_nodes"

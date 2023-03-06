from camera_manager import BaseCameraManager, CameraManager
from connection import BaseConnection, NTConnection
from magic_numbers import (
    COLOUR_AREA_TO_BOUNDING_BOX_THRESHOLD,
    CONE_HSV_HIGH,
    CONE_HSV_LOW,
    CUBE_HSV_HIGH,
    CUBE_HSV_LOW,
    CONE_HEIGHT,
    CUBE_HEIGHT,
    CONE_WIDTH,
    CUBE_WIDTH,
    camera_params1,
)

from math import atan2
import cv2
import numpy as np
from helper_types import (
    NodeView,
    Node,
    NodeObservation,
    GamePiece,
    BoundingBox,
)
from camera_config import CameraParams
from node_map import ALL_NODES
from wpimath.geometry import Pose2d, Pose3d, Translation3d, Transform3d
from typing import List

class GamePieceVision:
    def __init__(self, camera: BaseCameraManager, connection: BaseConnection) -> None:
        self.camera = camera
        self.connection = connection

        params = camera.get_params()
        self.robot_to_camera = params.transform
        self.camera_pose = Pose3d().transformBy(self.robot_to_camera)

    def run(self) -> None:
        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """
        frame_time, frame = self.camera.get_frame()
        # frame time is 0 in case of an error
        if frame_time == 0:
            self.camera.notify_error(self.camera.get_error())
            return
        
        robot_pose = self.connection.get_latest_pose()
        results, display = self.process_image(frame, robot_pose)

        if results is not None:
            # TODO send results to rio
            pass

        if self.connection.get_debug():
            # send image to display on driverstation
            self.camera.send_frame(display)

    def process_image(
        self, frame: np.ndarray, robot_pose: Pose2d
    ) -> tuple[list[NodeObservation], np.ndarray]:
        self.camera_pose = Pose3d(robot_pose).transformBy(self.robot_to_camera)
        visible_nodes = self.find_visible_nodes(frame, self.camera_pose)
        node_states = self.detect_node_state(frame, visible_nodes)

        push: List[str] = []
        for node_vision, state in zip(visible_nodes, node_states):
            push.append(
                node_vision.node.id.to_bytes(1, "big").hex()
                + list(state.occupied.to_bytes(1, "big").hex())[1]
            )
        self.connection.set_nodes(
            push or [""]
        )
        # annotate frame
        annotated_frame = self.annotate_image(frame, node_states)

        # return state of map (state of all node regions) and annotated camera stream
        return node_states, annotated_frame

    def is_coloured_game_piece(
        self,
        masked_image: np.ndarray,
        lower_colour: np.ndarray,
        upper_colour: np.ndarray,
    ) -> bool:
        gamepiece_mask = cv2.inRange(masked_image, lower_colour, upper_colour)
        return (
            np.count_nonzero(gamepiece_mask)
            > masked_image.shape[0]
            * masked_image.shape[1]
            * COLOUR_AREA_TO_BOUNDING_BOX_THRESHOLD
        )

    def is_game_piece_present(
        self,
        frame: np.ndarray,
        bounding_box: BoundingBox,
        expected_game_piece: GamePiece,
    ) -> bool:
        roi = frame[
            bounding_box.y1 : bounding_box.y2, bounding_box.x1 : bounding_box.x2
        ]
        if roi.size == 0:
            return False
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        cube_present = False
        cone_present = False
        if (
            expected_game_piece == GamePiece.BOTH
            or expected_game_piece == GamePiece.CUBE
        ):
            cube_present = self.is_coloured_game_piece(
                roi_hsv, CUBE_HSV_LOW, CUBE_HSV_HIGH
            )

        if (
            not cube_present and expected_game_piece == GamePiece.BOTH
        ) or expected_game_piece == GamePiece.CONE:
            cone_present = self.is_coloured_game_piece(
                roi_hsv, CONE_HSV_LOW, CONE_HSV_HIGH
            )

        return cone_present or cube_present

    def calculate_bounding_box(
        self,
        transl: Translation3d,
        node: Node,
        camera_pose: Pose3d,
        params: CameraParams,
    ) -> BoundingBox:
        """Determine appropriate bounding box based on location of game piece relative to camera"""

        is_cube = node.expected_game_piece == GamePiece.CUBE
        wmh = 0.5 * (CUBE_WIDTH if is_cube else CONE_WIDTH)
        hmh = 0.5 * (CUBE_HEIGHT if is_cube else CONE_HEIGHT)

        projected, _ = cv2.projectPoints(
            objectPoints=np.array(
                [
                    [-transl.y - wmh, -transl.z - hmh, transl.x],
                    [-transl.y + wmh, -transl.z + hmh, transl.x],
                ]
            ),
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            cameraMatrix=params.K,
            distCoeffs=None,
        )
        x1, y1 = projected[0][0]
        x2, y2 = projected[1][0]
        return BoundingBox(int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def get_camera_pose(pose: Pose2d, camera_params: CameraParams) -> Pose3d:
        """Return pose of camera-center in world's reference frame"""
        return Pose3d(pose) + camera_params.transform

    def find_visible_nodes(
        self, frame: np.ndarray, camera_pose: Pose3d
    ) -> list[NodeView]:
        """Segment image to find visible nodes in a frame

        Args:
            `frame` (np.ndarray): New camera frame containing nodes
            `camera_pose` (Pose3d): Current camera pose in the world frame

        Returns:
            `List[NodeView]`: List of node views with no information about occupancy
        """
        params = self.camera.get_params()
        nodes_in_camera = [node_to_camera(camera_pose, params, n) for n in ALL_NODES]
        candidates = [(n, t) for (n, t) in zip(ALL_NODES, nodes_in_camera) if t.x > 0.0]

        screen_bb = BoundingBox(0, 0, frame.shape[1], frame.shape[0])
        bbs = (
            self.calculate_bounding_box(t, n, self.camera_pose, params).intersection(
                screen_bb
            )
            for (n, t) in candidates
        )

        return [
            NodeView(bb, n) for (bb, (n, _)) in zip(bbs, candidates) if bb is not None
        ]

    def detect_node_state(
        self, frame: np.ndarray, regions_of_interest: list[NodeView]
    ) -> list[NodeObservation]:
        """Detect node occupancy in a set of observed node regions

        Args:
            frame (np.ndarray): New camera frame containing nodes
            regions_of_interest (List[NodeRegionObservation]): List of node region observations with no information about occupancy

        Returns:
            List[NodeRegionObservation]: List of node region observations
        """
        observations = []
        for view in regions_of_interest:
            occupied = self.is_game_piece_present(
                frame, view.bounding_box, view.node.expected_game_piece
            )
            observations.append(NodeObservation(view, occupied))
        return observations

    def annotate_image(
        self,
        frame: np.ndarray,
        observations: list[NodeObservation],
    ) -> np.ndarray:
        """annotate a frame with projected node points

        Args:
            frame (np.ndarray): raw image frame without annotation
            observations (List[NodeRegionObservation]): node observations in the current time step

        Returns:
            np.ndarray: frame annotated with observed node regions
        """
        for node in observations:
            col = (0, 255, 0) if node.occupied else (0, 0, 255)
            bb = node.view.bounding_box
            cv2.rectangle(frame, [bb.x1, bb.y1], [bb.x2, bb.y2], col, 2)
        return frame


def node_to_camera(
    camera_pose: Pose3d,
    camera_params: CameraParams,
    node: Node,
) -> Translation3d:
    world_to_camera = Transform3d(camera_pose, Pose3d())
    return (
        node.position.rotateBy(world_to_camera.rotation())
        + world_to_camera.translation()
    )


def is_node_in_image(
    camera_pose: Pose3d,
    camera_params: CameraParams,
    node: Node,
) -> bool:
    node_in_camera = node_to_camera(camera_pose, camera_params, node)
    return point3d_in_field_of_view(node_in_camera, camera_params)


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


def main():
    left_camera = CameraManager(
        0,
        "/dev/video0",
        camera_params1,
        "kYUYV",
    )

    vision = GamePieceVision(
        left_camera,
        NTConnection("left_cam"),
    )
    while True:
        vision.run()

if __name__ == "__main__":
    main()
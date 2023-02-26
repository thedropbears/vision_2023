from camera_manager import BaseCameraManager, CameraManager
from connection import BaseConnection, NTConnection
from magic_numbers import (
    CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD,
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
        print(f"seeing {len(node_states)} nodes from {self.camera_pose}")
        # annotate frame
        annotated_frame = self.annotate_image(frame, node_states)

        # return state of map (state of all node regions) and annotated camera stream
        return node_states, annotated_frame

    def is_coloured_game_piece(
        self,
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
        self,
        frame: np.ndarray,
        bounding_box: BoundingBox,
        expected_game_piece: GamePiece,
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
            expected_game_piece == GamePiece.BOTH
            or expected_game_piece == GamePiece.CUBE
        ):
            # run cube mask
            lower_purple = CUBE_HSV_LOW
            upper_purple = CUBE_HSV_HIGH
            cube_present = self.is_coloured_game_piece(
                hsv, lower_purple, upper_purple, bounding_box.area()
            )

        if (
            expected_game_piece == GamePiece.BOTH
            or expected_game_piece == GamePiece.CONE
        ):
            # run cone mask
            lower_yellow = CONE_HSV_LOW
            upper_yellow = CONE_HSV_HIGH

            cone_present = self.is_coloured_game_piece(
                hsv, lower_yellow, upper_yellow, bounding_box.area()
            )

        return cone_present or cube_present

    def calculate_bounding_box(
        self,
        centre: tuple[int, int],
        camera_pose: Pose3d,
        node: Node,
        params: CameraParams,
    ) -> BoundingBox:
        """Determine appropriate bounding box based on location of game piece relative to camera

        Args:
            `centre` (tuple): x and y coordinates of centre of node in image frame
            `camera_pose` (Translation3d): pose of the camera in the world frame
            `node` (Node) What node it is
            `camera_params` (CameraParams): relevant camera parameters for node region observation

        Returns:
            `BoundingBox`: bounding box within which a game piece is expected to be contained
        """

        is_cube = node.expected_game_piece == GamePiece.CUBE
        # Get max dimension of game piece
        gp_height_m = CUBE_HEIGHT if is_cube else CONE_HEIGHT
        gp_width_m = CUBE_WIDTH if is_cube else CONE_WIDTH

        dist = camera_pose.translation().distance(node.position)
        # Get gamepiece size in pixels
        gp_width = (gp_width_m / dist) * (params.width / params.get_fx()) * params.width
        gp_height = (
            (gp_height_m / dist) * (params.height / params.get_fy()) * params.height
        )

        x1 = int(centre[0] - gp_width / 2)
        y1 = int(centre[1] - gp_height / 2)
        x2 = int(centre[0] + gp_width / 2)
        y2 = int(centre[1] + gp_height / 2)

        # Check against bounds of image
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > params.width:
            x2 = params.width
        if y2 > params.height:
            y2 = params.height

        return BoundingBox(x1, y1, x2, y2)

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
            `pose` (Pose2d): Current robot pose in the world frame

        Returns:
            `List[NodeView]`: List of node views with no information about occupancy
        """
        params = self.camera.get_params()
        nodes_in_camera = [node_to_camera(camera_pose, params, n) for n in ALL_NODES]
        candidates = [(n, t) for (n, t) in zip(ALL_NODES, nodes_in_camera) if t.x > 0.0]
        candidate_nodes, candidate_transls = zip(*candidates)

        projected, _ = cv2.projectPoints(
            objectPoints=np.array(
                [[-t.y, -t.z, t.x] for t in candidate_transls]
            ),  # Opencv wants Z forward, we have X
            rvec=np.array([0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            cameraMatrix=params.K,
            distCoeffs=None,
        )
        screen_bb = BoundingBox(0, 0, frame.shape[1], frame.shape[0])
        bbs = (
            self.calculate_bounding_box(c, self.camera_pose, n, params).intersection(
                screen_bb
            )
            for (c, n) in zip(projected[:, 0], candidate_nodes)
        )

        return [
            NodeView(bb, n) for (bb, n) in zip(bbs, candidate_nodes) if bb is not None
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
            map (List[NodeState]): current map state with information on occupancy state_
            node_observations (List[NodeRegionObservation]): node observations in the current time step

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


if __name__ == "__main__":
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

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
    def __init__(self, cameras: list[CameraManager], connection: NTConnection) -> None:
        self.cameras = cameras
        self.connection = connection
        self.map = NodeRegionMap()

    def run(self) -> None:
        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """
        self.connection.pong()
        for camera_manager in self.cameras:
            frame_time, frame = camera_manager.get_frame()
            # frame time is 0 in case of an error
            if frame_time == 0:
                camera_manager.notify_error(camera_manager.get_error())
                return

            results, display = self.process_image(frame)

            if results is not None:
                # TODO send results to rio
                pass

            # send image to display on driverstation
            camera_manager.send_frame(display)
            
        self.connection.set_fps()


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
        expected_game_piece: ExpectedGamePiece
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


    def process_image(self, frame: np.ndarray, pose: Pose2d):

        # visible_nodes = self.find_visible_nodes(frame, pose)

        # node_states = self.detect_node_state(frame, visible_nodes)

        # whatever the update step is
        # self.map.update(node_states)

        # map_state = self.map.get_state()

        # annotate frame
        # annotated_frame = annotate_image(frame, map_state, node_states)

        # return state of map (state of all node regions) and annotated camera stream
        return


    def calculate_bounding_box(
        self,
        centre: tuple[int, int],
        node_point: Translation3d,
        expected_game_piece: ExpectedGamePiece,
        camera_params: CameraParams
        ) -> BoundingBox:
        """Determine appropriate bounding box based on location of game piece relative to camera

        Args:
            centre (tuple): x and y coordinates of centre of node in image frame
            node_point (Translation3d): pose of node in camera frame
            camera_params (CameraParams): relevant camera parameters for node region observation

        Returns:
            BoundingBox: bounding box within which a game piece is expected to be contained
        """

        # Get max dimension of game piece
        if expected_game_piece == ExpectedGamePiece.CUBE:
            gp_size = CUBE_HEIGHT
        else:
            gp_size = CONE_HEIGHT

        # Get gamepiece size in pixels
        # TODO: convert gp size to pixels
        # gp_width

        x1 = centre(0) - gp_width / 2
        y1 = centre(1) - gp_width / 2
        x2 = centre(0) + gp_width / 2
        y2 = centre(1) + gp_width / 2

        # Check against bounds of image
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > camera_params.width:
            x2 = camera_params.width
        if y2 > camera_params.height:
            y2 = camera_params.height

        return BoundingBox(x1, y1, x2, y2)


    def find_visible_nodes(self, frame: np.ndarray, pose: Pose2d) -> list[NodeRegionObservation]:
        """Segment image to find visible nodes in a frame

        Args:
            frame (np.ndarray): New camera frame containing nodes
            pose (Pose2d): Current robot pose in the world frame

        Returns:
            List[NodeRegionObservation]: List of node region observations with no information about occupancy
        """

        visible_nodes: list[NodeRegionObservation] = []

        # Find visible nodes from node map
        for node_state in self.map.get_state():
            # Check if node region is visble in any camera
            for camera_manager in self.cameras:
                params = camera_manager.get_params()
                if self.is_node_region_in_image(robot_pose, params, node_state.node_region):        
                    # create transform to make camera origin
                    # TODO: make transform pose a function
                    world_to_robot = Transform3d(Pose3d(), Pose3d(robot_pose))
                    world_to_camera = world_to_robot + camera_params.transform
                    node_region_camera_frame = (
                        Pose3d(node_region.position, Rotation3d()) + world_to_camera.inverse()
                    )
                    node_position = node_region_camera_frame.translation()

                    # Get image coordinates of centre of node region
                    x_coord = params.width/2 + node_position.y * params.get_fx() / node_position.x 
                    y_coord = params.height/2 + node_position.z * params.get_fy() / node_position.x
                    # TODO: distort point as above coordinates are calculated for projection into undistorted image frame
                    coords = (int(x_coord), int(y_coord))

                    # Calculate bounding box
                    bb = self.calculate_bounding_box(coords, node_position, params)
                    visible_nodes.append(NodeRegionObservation(camera_manager.get_id(), bb, node_state.node_region))

        return visible_nodes


    def detect_node_state(
        self,
        frame: np.ndarray,
        regions_of_interest: list[NodeRegionObservation]
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
        self,
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
        self,
        robot_pose: Pose2d,
        camera_params: CameraParams,
        node_region: NodeRegion,
    ) -> bool:

        # create transform to make camera origin
        world_to_robot = Transform3d(Pose3d(), Pose3d(robot_pose))
        world_to_camera = world_to_robot + camera_params.transform
        node_region_camera_frame = (
            Pose3d(node_region.position, Rotation3d()) + world_to_camera.inverse()
        )

        # Check the robot is facing the right direction for the point by checking it is inside the FOV
        return self.point3d_in_field_of_view(
            node_region_camera_frame.translation(), camera_params
        )


    def point3d_in_field_of_view(self, point: Translation3d, camera_params: CameraParams) -> bool:
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

    cameras: list[CameraManager] = []
    left_camera = CameraManager(
            0,
            "/dev/video0",
            params,
            "kYUYV",
    )
    cameras.append(left_camera)
    #  Setup camera managers for any other cameras and append to 'cameras'

    vision = Vision(
        cameras,
        NTConnection(),
    )
    while True:
        vision.run()

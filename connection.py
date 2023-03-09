import math
import time
from wpimath.geometry import Pose2d
from networktables import NetworkTablesInstance, NetworkTables
from abc import ABC, abstractmethod
from typing import Optional


RIO_IP = {True: "127.0.0.1", False: "10.47.74.2"}


def nt_data_to_node_data(self, data: list[str]) -> list[tuple[int, bool]]:
    nodes: list[tuple[int, bool]] = []
    for node in data:
        as_array = str(node)
        a = (int(f"{as_array[0]}{as_array[1]}"), as_array[2] == "1")
        nodes.append(a)
    return nodes


class BaseConnection(ABC):
    @abstractmethod
    def send_results(self, positive: list[int], negatives: list[int]) -> None:
        ...

    @abstractmethod
    def get_latest_pose(self) -> Pose2d:
        ...

    @abstractmethod
    def get_debug(self) -> bool:
        ...

    @abstractmethod
    def set_nodes(self, value: list[str]) -> None:
        ...


class NTConnection(BaseConnection):
    inst: NetworkTablesInstance

    def __init__(
        self, name: str, inst: NetworkTablesInstance = NetworkTables, sim: bool = False
    ) -> None:
        inst.initialize(server=RIO_IP[sim])
        self.inst = inst

        nt = self.inst.getTable(name)
        self.nodes_entry = nt.getEntry("nodes")
        self.true_entry = nt.getEntry("results_true")
        self.false_entry = nt.getEntry("results_false")
        self.timestamp_entry = nt.getEntry("timestamp")
        self.fps_entry = nt.getEntry("fps")
        # wether to stream an annotated image back
        self.debug_entry = nt.getEntry("debug_stream")
        self.debug_entry.setBoolean(False)

        pose_table = self.inst.getTable("SmartDashboard").getSubTable("Field")
        self.pose_entry = pose_table.getEntry("fused_pose")

        self.old_fps_time = 0.0
        self._get_time = time.monotonic

    def send_results(self, positives: list[int], negatives: list[int]) -> None:
        # self.true_entry.setIntegerArray(positives)
        # self.false_entry.setIntegerArray(negatives)

        current_time = self._get_time()
        self.timestamp_entry.setDouble(current_time)
        fps = 1 / (current_time - self.old_fps_time)
        self.old_fps_time = current_time
        self.fps_entry.setDouble(fps)

        self.inst.flush()

    def set_nodes(self, value: list[str]) -> None:
        self.nodes_entry.setStringArray(value)

        current_time = self._get_time()
        self.timestamp_entry.setDouble(current_time)
        fps = 1 / (
            (current_time - self.old_fps_time)
            if (current_time - self.old_fps_time) != 0
            else 1
        )
        self.old_fps_time = current_time
        self.fps_entry.setDouble(fps)

        self.inst.flush()

    def get_latest_pose(self) -> Pose2d:
        arr: list[float] = self.pose_entry.getDoubleArray([0, 0, 0]) # type: ignore
        print(arr)
        return Pose2d(arr[0], arr[1], math.radians(arr[2]))

    def get_debug(self) -> bool:
        # return self.debug_entry.getBoolean(True)
        return True


zero_pose = Pose2d()


class DummyConnection(BaseConnection):
    def __init__(self, pose=zero_pose, do_print=False, do_annotate=False):
        self.pose = pose
        self.debug = do_annotate
        self.results = [[], []]
        self.string_array = []

    def send_results(self, positives: list[int], negatives: list[int]) -> None:
        self.results = [positives, negatives]
        print(
            "results being sent, positive sightings:",
            positives,
            ", negative signtings:",
            negatives,
        )

    def get_latest_pose(self) -> Pose2d:
        return self.pose

    def get_debug(self) -> bool:
        return self.debug

    def set_nodes(self, value: list[str]) -> None:
        self.string_array = value
        print(f"Setting nodes to {value}")

import math
import time
from wpimath.geometry import Pose2d
from networktables import NetworkTablesInstance
from abc import ABC, abstractmethod

NetworkTables = NetworkTablesInstance.getDefault()

RIO_IP = "127.0.0.1" #"10.47.74.2"


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


class NTConnection(BaseConnection):
    def __init__(self, name: str, inst: NetworkTablesInstance = NetworkTables) -> None:
        inst.initialize(server=RIO_IP)
        self.inst = inst

        nt = inst.getTable("/vision" + name)
        self.true_entry = nt.getEntry("results_true")
        self.false_entry = nt.getEntry("results_false")
        self.timestamp_entry = nt.getEntry("timestamp")
        self.fps_entry = nt.getEntry("fps")
        # wether to stream an annotated image back
        self.debug_entry = nt.getEntry("debug_stream")
        self.debug_entry.setBoolean(False)

        pose_table = inst.getTable("/SmartDashboard/Field")
        self.pose_entry = pose_table.getEntry("fused_pose")

        self.old_fps_time = 0.0
        self._get_time = time.monotonic

    def send_results(self, positives: list[int], negatives: list[int]) -> None:
        self.true_entry.setDoubleArray(positives)
        self.false_entry.setDoubleArray(negatives)

        current_time = self._get_time()
        self.timestamp_entry.setDouble(current_time)
        fps = 1 / (current_time - self.old_fps_time)
        self.old_fps_time = current_time
        self.fps_entry.setDouble(fps)

        self.inst.flush()

    def get_latest_pose(self) -> Pose2d:
        arr = self.pose_entry.getDoubleArray([0, 0, 0])
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

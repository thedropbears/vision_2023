import numpy as np
import cv2
from camera_config import CameraParams
import sys
from abc import ABC, abstractmethod


class BaseCameraManager(ABC):
    @abstractmethod
    def get_params(self) -> CameraParams:
        ...

    @abstractmethod
    def get_frame(self) -> tuple[int, np.ndarray]:
        ...

    @abstractmethod
    def send_frame(self, frame: np.ndarray) -> None:
        ...

    @abstractmethod
    def get_error(self) -> str:
        ...

    @abstractmethod
    def notify_error(self, error: str) -> None:
        ...


class CameraManager(BaseCameraManager):
    def __init__(
        self,
        camera_id: int,
        path: str,
        params: CameraParams,
        pixel_format: str,
    ) -> None:
        """Initialises a Camera Manager

        Args:
            path: The path of the camera (can be id, path, or /dev/video)
            params: Specific parameters for the camera including intrinsics, image dimensions etc...
            pixel_format: The video's pixel format (kYUYV, kMJPEG, etc)
        """
        from cscore import CameraServer, VideoMode

        self.camera_id = camera_id

        self.params = params

        self.camera = CameraServer.startAutomaticCapture(name=params.name, path=path)
        self.camera.setVideoMode(
            getattr(VideoMode.PixelFormat, pixel_format),
            params.width,
            params.height,
            params.fps,
        )

        # In this, source and sink are inverted from the cscore documentation.
        # self.sink is a CvSource and self.sources are CvSinks. This is because it makes more sense for a reader.
        # We get images from a source, and put images to a sink.
        self.source = CameraServer.getVideo(camera=self.camera)
        self.sink = CameraServer.putVideo("Driver_Stream", params.width, params.height)
        # Width and Height are reversed here because the order of putVideo's width and height
        # parameters are the opposite of numpy's (technically it is an array, not an actual image).
        self.frame = np.zeros(shape=(params.height, params.width, 3), dtype=np.uint8)

        self.set_camera_property("white_balance_temperature_auto", 0)
        self.set_camera_property("exposure_auto_priority", 0)
        self.set_camera_property("exposure_auto", 1)
        self.set_camera_property("focus_auto", 0)
        self.set_camera_property("exposure_absolute", 10)
        self.set_camera_property("raw_contrast", 255)
        self.set_camera_property("contrast", 100)
        self.set_camera_property("raw_saturation", 255)
        self.set_camera_property("saturation", 100)
        self.set_camera_property("raw_gain", 102)
        self.set_camera_property("gain", 40)
        self.set_camera_property("white_balance_temperature", 6500)
        self.set_camera_property("brightness", 50)
        self.set_camera_property("raw_brightness", 127)

    def get_id(self) -> int:
        """Get the unique camera id
        Returns:
            int
        """
        return self.camera_id

    def get_params(self) -> CameraParams:
        """Gets the camera parameters of the associated camera
        Returns:
            CameraParams
        """
        return self.params

    def get_frame(self) -> tuple[int, np.ndarray]:
        """Gets a frame from the camera.
        Returns:
            Time the frame was captured in microseconds, or 0 on error.
            A numpy array of the frame, dtype=np.uint8, BGR.
        """
        frame_time, self.frame = self.source.grabFrameNoTimeout(image=self.frame)
        return frame_time, self.frame

    def send_frame(self, frame: np.ndarray) -> None:
        """Sends a frame to the driver display.
        Args:
            frame: A numpy array image. (Should always be the same size)
        """
        self.sink.putFrame(frame)

    def get_error(self) -> str:
        """Gets an error from the camera.
        Should be run by Vision when frame_time is 0.
        Returns:
            A string containing the camera's error.
        """
        return self.source.getError()

    def notify_error(self, error: str) -> None:
        """Sends an error to the console and the sink.
        Args:
            error: The string to send. Should be gotten by get_error().
        """
        print(error, file=sys.stderr)
        self.sink.notifyError(error)

    def set_camera_property(self, property: str, value: int) -> None:
        self.camera.getProperty(property).set(value)


class MockImageManager(BaseCameraManager):
    def __init__(
        self, image: np.ndarray, params: CameraParams, display_output: bool = False
    ) -> None:
        """Initialises a Mock Image Manager
        Args:
            image: A BGR numpy image array
        """
        self.image = image
        self.display_output = display_output
        self.params = params

    def change_image(self, new_image: np.ndarray) -> None:
        """Changes self.image.
        Args:
            new_image: The new image to switch to. Should be a numpy image array.
        """
        self.image = new_image

    def get_frame(self) -> tuple[int, np.ndarray]:
        """Returns self.image.
        Returns:
            1: Simulates the frame_time
            self.image, a BGR numpy array.
        """
        return 1, self.image.copy()

    def send_frame(self, frame: np.ndarray):
        if self.display_output:
            cv2.imshow("Image", frame)
            cv2.waitKey(0)

    def get_error(self) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.
        """
        print(error, file=sys.stderr)

    def get_params(self) -> CameraParams:
        return self.params


class MockVideoManager(BaseCameraManager):
    def __init__(self, video: cv2.VideoCapture, display_output: bool = False):
        """Initialises a Mock Video Manager.
        Args:
            video: An opencv video, as received by cv2.VideoCapture
        """
        self.video = video
        self.display_output = display_output

    def get_frame(self) -> tuple[int, np.ndarray]:
        """Returns the next frame of self.video.
        Returns:
            Whether or not it was successful. False means error.
            The next frame of self.video.
        """
        result = self.video.read()
        if result[0]:
            return result
        else:  # If we reach the end of the video, go back to the beginning.
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.video.read()

    def send_frame(self, frame: np.ndarray) -> None:
        if self.display_output:
            cv2.imshow("Image", frame)
            cv2.waitKey(0)

    def get_error(self) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.
        """
        print(error, file=sys.stderr)


class WebcamCameraManager(BaseCameraManager):
    def __init__(self, camera: int, params: CameraParams) -> None:
        """Initialises a Webcam Camera Manager. Designed to run on a non-pi computer.
        Initialises it with the first detected system camera, for example a webcam.

        Args:
            camera: Which camera to use. Default is 0th, probably a builtin webcam for most people.
        """
        self.video = cv2.VideoCapture(camera)
        self.params = params

    def get_frame(self) -> tuple[int, np.ndarray]:
        """Returns the current video frame.
        Returns:
            Whether or not it was successful. False means error.
            The current video frame.
        """
        return self.video.read()

    def send_frame(self, frame: np.ndarray) -> None:
        cv2.imshow("image", frame)
        cv2.waitKey(1)

    def get_error(self) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.
        """
        print(error, file=sys.stderr)

    def get_params(self) -> CameraParams:
        return self.params

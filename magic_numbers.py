import numpy as np

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
CONE_HSV_LOW = np.array((0, 0, 0), dtype=np.uint8)
CONE_HSV_HIGH = np.array((180, 255, 255), dtype=np.uint8)

CUBE_HSV_LOW = np.array((0, 0, 0), dtype=np.uint8)
CUBE_HSV_HIGH = np.array((180, 255, 255), dtype=np.uint8)

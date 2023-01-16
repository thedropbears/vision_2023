import numpy as np

# Camera settings
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Target HSV for masking
CONE_HSV_LOW = np.array([(28 / 240) * 180, (118 / 240) * 255, (107 / 240) * 255])
CONE_HSV_HIGH = np.array([(35 / 240) * 180, (240 / 240) * 255, (220 / 240) * 255])

CUBE_HSV_LOW = np.array([(160 / 240) * 180, (99 / 240) * 255, (59 / 240) * 255])
CUBE_HSV_HIGH = np.array([(185 / 240) * 180, (240 / 240) * 255, (225 / 240) * 255])

CONTOUR_TO_BOUNDING_BOX_AREA_RATIO_THRESHOLD = 0.1

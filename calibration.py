import numpy as np
import cv2 as cv
import glob
import sys


gridrows = 9
gridcols = 6
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((gridrows * gridcols, 3), np.float32)
objp[:, :2] = np.mgrid[0:gridrows, 0:gridcols].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob("chessboard_images/*.jpg")


nr_total = 0
nr_success = 0
video = cv.VideoCapture(
    "chessboard_videos/WIN_20230121_12_00_08_microsoft_a3chessboard.mp4"
)

# fps = video.get(cv.CAP_PROP_FPS)
# minutes = 0
# seconds = 0
# frame_id = int(fps * (minutes * 60 + seconds))
# video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
total_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
print(total_frames)
current_frame = 0


while video.isOpened():
    # img = cv.imread(fname)

    if current_frame >= total_frames:
        break

    is_read, frame = video.read()
    current_frame += 1
    if not is_read:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("frame", gray)
    # cv.waitKey(1)
    ret, corners = cv.findChessboardCorners(
        gray, (gridcols, gridrows), flags=cv.CALIB_CB_ADAPTIVE_THRESH
    )
    # print(corners)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(frame, (gridrows, gridcols), corners2, ret)
        cv.imshow("img", frame)
        cv.waitKey(1)
        # print(f"saw chessboard")
        nr_success += 1
    else:
        print(f"failed to see chessboard")


print("Done")
cv.destroyAllWindows()

print("Calibrating Camera")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)


# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the frame
# x, y, w, h = roi
# dst = dst[y : y + h, x : x + w]

# cv.imshow("ahh", img)
# cv.waitKey(1)
print(f"successful images {nr_success}")
print(f"camera matrix: \n {mtx}")
print(f"distortion coefficients: \n {dist}")
# print(ret, mtx, dist, rvecs, tvecs)
# cv.destroyAllWindows()

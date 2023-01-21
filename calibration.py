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
video = cv.VideoCapture("chessboard_videos/LogiC920_a3_chessboard_0630.mp4")

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

    frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("frame", gray)
    # cv.waitKey(1)
    ret, corners = cv.findChessboardCorners(gray, (gridcols, gridrows), None)
    # print(corners)

    if ret == True:
        if current_frame % 10 == 0:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(frame, (gridcols, gridrows), corners2, ret)
            cv.imshow("img", frame)
            # cv.waitKey(1)
            # print(f"saw chessboard")
            nr_success += 1
    else:
        print(f"failed to see chessboard")
    cv.waitKey(1)


print("Done")
cv.destroyAllWindows()

print("Calibrating Camera")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

h, w = frame.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
# cv.imwrite('calibresult.png', dst)
cv.imshow("dst", dst)
cv.waitKey(0)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

# cv.imshow("ahh", img)
# cv.waitKey(1)
print(f"successful images {nr_success}")
print(f"camera matrix: \n {mtx}")
print(f"distortion coefficients: \n {dist}")
# print(ret, mtx, dist, rvecs, tvecs)
# cv.destroyAllWindows()

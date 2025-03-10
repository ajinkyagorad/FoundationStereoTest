import cv2
import numpy as np
import json
import glob

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Step 1: Capture calibration images
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

image_count = 0
while image_count < 10:  # Collect 10 images for calibration
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(500)  # Wait for 500ms before taking next image
        image_count += 1

cap.release()
cv2.destroyAllWindows()

# Perform calibration
if len(objpoints) < 10:
    print("Not enough images captured for calibration. Try again.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save calibration results
calibration_data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
with open('camera_calibration.json', 'w') as f:
    json.dump(calibration_data, f)

print("Calibration complete. Coefficients saved to 'camera_calibration.json'")
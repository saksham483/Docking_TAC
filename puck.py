import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================

MARKER_SIZE = 0.15 
X_OFF = 0.29
Y_OFF = 0.49

# We define where the markers are PHYSICALLY located relative to the center (0,0,0)
# Based on your previous vectors:
# If Marker 28 vector_to_center is [0.29, -0.49], its position is [-0.29, 0.49]
MARKER_POSITIONS = {
    28: np.array([-X_OFF,  Y_OFF, 0], dtype=np.float32), # Top-Left
    7:  np.array([ X_OFF,  Y_OFF, 0], dtype=np.float32), # Top-Right
    19: np.array([-X_OFF, -Y_OFF, 0], dtype=np.float32), # Bottom-Left
    96: np.array([ X_OFF, -Y_OFF, 0], dtype=np.float32)  # Bottom-Right
}

CALIB_FILE = 'calibration_data.json'
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
PARAMS = aruco.DetectorParameters_create()
PARAMS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

# ==========================================
# 2. BOARD CREATION
# ==========================================
def create_custom_board():
    # We need to define the 4 corners of every marker relative to the board center
    obj_points = []
    ids = []
    
    half_s = MARKER_SIZE / 2.0
    
    # Standard ArUco Corner Order: TopLeft, TopRight, BottomRight, BottomLeft
    # We create the square shape around the marker center
    base_square = np.array([
        [-half_s,  half_s, 0], 
        [ half_s,  half_s, 0],
        [ half_s, -half_s, 0],
        [-half_s, -half_s, 0]
    ], dtype=np.float32)

    for marker_id, center_pos in MARKER_POSITIONS.items():
        # Add the center offset to the base square corners
        corners = base_square + center_pos
        obj_points.append(corners)
        ids.append(marker_id)

    # Create the Board object
    board = aruco.Board_create(np.array(obj_points), ARUCO_DICT, np.array(ids))
    return board

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def load_calibration(path):
    if not os.path.exists(path):
        print(f"Error: Calibration file '{path}' not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    dist_key = "dist_coeff" if "dist_coeff" in data else "dist_coeffs"
    return np.array(data['camera_matrix'], dtype=np.float32), \
           np.array(data[dist_key], dtype=np.float32)

# ==========================================
# 4. MAIN LOOP
# ==========================================

def main():
    mtx, dist = load_calibration(CALIB_FILE)
    board = create_custom_board()
    cap = cv2.VideoCapture(0)

    print("Robust Board Tracker Running...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=PARAMS)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # ---------------------------------------------------------
            # THE ROBUST METHOD: Estimate Pose of the BOARD (Global Optimization)
            # ---------------------------------------------------------
            # This function uses all visible corners to find the one true center.
            valid, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist, None, None)

            if valid > 0: # If at least one marker is found and pose calculated
                
                # 1. Draw the robust Board Axis at (0,0,0)
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.2)
                
                # 2. Draw the robust Center Dot
                imgpts, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, mtx, dist)
                center_screen = tuple(imgpts[0].ravel().astype(int))
                cv2.circle(frame, center_screen, 8, (0, 0, 255), -1) # Red Dot

                # 3. Draw Info
                t_val = tvec.flatten()
                text = f"Robust Center: {t_val[0]:.2f}, {t_val[1]:.2f}, {t_val[2]:.2f}"
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # -----------------------------------------------------
                # VISUALIZATION: Show individual contribution lines
                # (This shows you how much "drift" individual markers have compared to the board)
                # -----------------------------------------------------
                
                # Calculate Single Marker Poses just for drawing the yellow lines
                single_rvecs, single_tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
                
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in MARKER_POSITIONS:
                        # Where is the marker on the screen?
                        c_marker = tuple(corners[i][0].mean(axis=0).astype(int))
                        
                        # Where is the Robust Center on the screen? (center_screen)
                        # Draw line from Marker to Robust Center
                        cv2.line(frame, c_marker, center_screen, (0, 255, 255), 1)

        cv2.imshow('Robust Board Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

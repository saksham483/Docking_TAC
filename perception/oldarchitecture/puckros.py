#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import sys
from scipy.spatial.transform import Rotation as R, Slerp

# ==========================================
# 1. CLASS: ADAPTIVE POSE FILTER
# ==========================================

class AdaptivePoseFilter:
    """
    Adaptive Low Pass Filter.
    - High smoothing (low alpha) when robot is stationary (kills jitter).
    - Low smoothing (high alpha) when robot is moving (kills latency).
    """
    def __init__(self, min_alpha=0.1, max_alpha=0.9, ramp_dist=0.05):
        self.min_alpha = min_alpha  # Heavy smoothing
        self.max_alpha = max_alpha  # Trust raw data
        self.ramp_dist = ramp_dist  # Distance (m) to transition between smooth and raw
        self.prev_pos = None
        self.prev_quat = None

    def update(self, curr_pos, curr_quat):
        if self.prev_pos is None:
            self.prev_pos = curr_pos
            self.prev_quat = curr_quat
            return curr_pos, curr_quat

        # 1. Calculate how fast the pose is changing (Euclidean distance)
        dist = np.linalg.norm(curr_pos - self.prev_pos)
        
        # 2. Calculate dynamic alpha
        # If dist is 0, alpha = min_alpha (smooth). 
        # If dist > ramp_dist, alpha = max_alpha (responsive).
        alpha = np.clip((dist / self.ramp_dist), self.min_alpha, self.max_alpha)

        # 3. Position Filter (Linear Interpolation)
        filt_pos = alpha * curr_pos + (1 - alpha) * self.prev_pos

        # 4. Rotation Filter (SLERP)
        # We blend the previous rotation to the current one based on alpha
        key_times = [0, 1]
        key_rots = R.from_quat([self.prev_quat, curr_quat])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha]) 
        filt_quat = interp_rot[0].as_quat()

        # Update State
        self.prev_pos = filt_pos
        self.prev_quat = filt_quat
        
        return filt_pos, filt_quat

# ==========================================
# 2. BOARD CONFIGURATION
# ==========================================

X_OFF = 0.29
Y_OFF = 0.49

# Physical positions of marker centers relative to board center
MARKER_POSITIONS = {
    28: np.array([-X_OFF,  Y_OFF, 0], dtype=np.float32), # Top-Left
    7:  np.array([ X_OFF,  Y_OFF, 0], dtype=np.float32), # Top-Right
    19: np.array([-X_OFF, -Y_OFF, 0], dtype=np.float32), # Bottom-Left
    96: np.array([ X_OFF, -Y_OFF, 0], dtype=np.float32)  # Bottom-Right
}

# ==========================================
# 3. ROS 2 NODE
# ==========================================

class RobustDockingNode(Node):
    def __init__(self):
        super().__init__('robust_docking_node')
        
        # --- Parameters ---
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('calibration_file', 'calibration_data.json')
        self.declare_parameter('marker_size', 0.15)
        self.declare_parameter('video_device', 0)
        self.declare_parameter('enable_gui', True) 

        self.camera_frame = self.get_parameter('camera_frame').value
        self.calib_file = self.get_parameter('calibration_file').value
        self.marker_size = self.get_parameter('marker_size').value
        self.video_device = self.get_parameter('video_device').value
        self.enable_gui = self.get_parameter('enable_gui').value

        # --- Setup ---
        self.mtx, self.dist = self.load_calibration(self.calib_file)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        
        # Detector Optimization: Faster Subpix refinement
        self.params = aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 3 # Smaller window = Faster
        self.params.cornerRefinementMaxIterations = 20
        
        self.board = self.create_custom_board()
        self.filter = AdaptivePoseFilter(min_alpha=0.1, max_alpha=0.9, ramp_dist=0.05)

        # Tracking State
        self.last_rvec = None
        self.last_tvec = None

        # Camera Setup
        self.cap = cv2.VideoCapture(self.video_device)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Failed to open device {self.video_device}")
            sys.exit(1)
            
        # Hardware Latency Optimization
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

        # QoS for fast updates
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(PoseStamped, 'dock_pose', qos)

        # Run at 30Hz
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info("Robust Board Tracker Initialized (Adaptive Filter Active)")

    def load_calibration(self, path):
        if not os.path.exists(path):
            self.get_logger().fatal(f"Calibration file not found: {path}")
            sys.exit(1)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            dist_key = "dist_coeff" if "dist_coeff" in data else "dist_coeffs"
            return np.array(data['camera_matrix'], dtype=np.float32), \
                   np.array(data[dist_key], dtype=np.float32)
        except Exception as e:
            self.get_logger().fatal(f"Calibration Error: {e}")
            sys.exit(1)

    def create_custom_board(self):
        obj_points = []
        ids = []
        half_s = self.marker_size / 2.0
        
        base_square = np.array([
            [-half_s,  half_s, 0], 
            [ half_s,  half_s, 0],
            [ half_s, -half_s, 0],
            [-half_s, -half_s, 0]
        ], dtype=np.float32)

        for marker_id, center_pos in MARKER_POSITIONS.items():
            corners = base_square + center_pos
            obj_points.append(corners)
            ids.append(marker_id)

        return aruco.Board_create(np.array(obj_points), self.aruco_dict, np.array(ids))

    def timer_callback(self):
        # --- 1. BUFFER DRAIN (Latency Optimization) ---
        # Grab frame repeatedly to ensure we have the absolute latest image
        # This prevents the robot from reacting to what happened 200ms ago
        ret, frame = self.cap.read()
        if not ret: return

        # Optional: aggressive drain if camera fps > processing fps
        # while self.cap.grab():
        #     ret, frame = self.cap.retrieve()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is not None and len(ids) > 0:
            
            # --- 2. ROBUST ESTIMATION (With Stability Check) ---
            # Use previous frame as guess to prevent axis flipping
            use_guess = (self.last_rvec is not None)
            
            valid_markers, rvec, tvec = aruco.estimatePoseBoard(
                corners, ids, self.board, self.mtx, self.dist, 
                self.last_rvec, self.last_tvec, useExtrinsicGuess=use_guess
            )

            if valid_markers > 0:
                # --- 3. JUMP DETECTION (Anti-Teleport) ---
                # If pose jumps > 0.4m instantly, it's likely a solver error. Reset.
                if use_guess:
                    jump_dist = np.linalg.norm(tvec.flatten() - self.last_tvec.flatten())
                    if jump_dist > 0.4:
                        self.get_logger().warn(f"Impossible jump detected ({jump_dist:.2f}m). Resetting solver.")
                        self.last_rvec = None
                        self.last_tvec = None
                        return # Skip this bad frame

                # Update tracking
                self.last_rvec = rvec.copy()
                self.last_tvec = tvec.copy()

                # --- 4. PREPARE DATA ---
                raw_pos = tvec.flatten()
                rmat, _ = cv2.Rodrigues(rvec)
                raw_quat = R.from_matrix(rmat).as_quat()

                # --- 5. ADAPTIVE FILTERING ---
                filt_pos, filt_quat = self.filter.update(raw_pos, raw_quat)

                # --- 6. PUBLISH ---
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_frame
                msg.pose.position.x = float(filt_pos[0])
                msg.pose.position.y = float(filt_pos[1])
                msg.pose.position.z = float(filt_pos[2])
                msg.pose.orientation.x = float(filt_quat[0])
                msg.pose.orientation.y = float(filt_quat[1])
                msg.pose.orientation.z = float(filt_quat[2])
                msg.pose.orientation.w = float(filt_quat[3])
                self.publisher_.publish(msg)

                # --- 7. VISUALIZATION ---
                if self.enable_gui:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Draw Board Axis
                    cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 0.2)
                    
                    # Calculate Center Screen Point
                    imgpts, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, self.mtx, self.dist)
                    center_screen = tuple(imgpts[0].ravel().astype(int))
                    
                    # Draw Center Dot (Red)
                    cv2.circle(frame, center_screen, 6, (0, 0, 255), -1)

                    # Draw Yellow Lines (Individual Markers -> Board Center)
                    for i, marker_id in enumerate(ids.flatten()):
                        if marker_id in MARKER_POSITIONS:
                            c_marker = tuple(corners[i][0].mean(axis=0).astype(int))
                            cv2.line(frame, c_marker, center_screen, (0, 255, 255), 2)

                    # Info Text
                    text = f"X:{filt_pos[0]:.2f} Y:{filt_pos[1]:.2f} Z:{filt_pos[2]:.2f}"
                    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                # Tracking lost this frame
                self.last_rvec = None
                self.last_tvec = None

        if self.enable_gui:
            cv2.imshow('Robust Docking', frame)
            if cv2.waitKey(1) == 27:
                rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RobustDockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

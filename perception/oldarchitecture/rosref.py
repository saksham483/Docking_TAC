#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import sys
from scipy.spatial.transform import Rotation as R, Slerp

# ==========================================
# 1. HELPER CLASS: POSE FILTERING
# ==========================================

class PoseFilter:
    """
    Implements an Exponential Moving Average (Low Pass Filter) 
    to smooth out jittery ArUco detections.
    """
    def __init__(self, alpha_pos=0.7, alpha_rot=0.7):
        # Alpha 1.0 = No filtering (Raw data)
        # Alpha 0.1 = Heavy filtering (Slow response, very smooth)
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.prev_pos = None
        self.prev_quat = None

    def update(self, curr_pos, curr_quat):
        if self.prev_pos is None:
            self.prev_pos = curr_pos
            self.prev_quat = curr_quat
            return curr_pos, curr_quat

        # Filter Position (Linear Interpolation)
        filt_pos = self.alpha_pos * curr_pos + (1 - self.alpha_pos) * self.prev_pos

        # Filter Rotation (Spherical Linear Interpolation - SLERP)
        # We use scipy to SLERP between the previous quaternion and the new one
        key_times = [0, 1]
        key_rots = R.from_quat([self.prev_quat, curr_quat])
        slerp = Slerp(key_times, key_rots)
        # Interpolate at time 1.0 using the alpha weight conceptually
        # Note: A simple SLERP implementation for EMA:
        # q_new = Slerp(q_old, q_meas, t=alpha)
        
        # Re-creating slerp specifically for EMA logic:
        # We blend the previous (time 0) to current (time 1) by factor alpha
        interp_rot = slerp([self.alpha_rot]) 
        filt_quat = interp_rot[0].as_quat()

        # Update state
        self.prev_pos = filt_pos
        self.prev_quat = filt_quat
        
        return filt_pos, filt_quat

# ==========================================
# 2. DEFAULT MARKER MAP
# ==========================================

DEFAULT_MARKER_MAP = {
    28: [ 0.29, -0.49, 0.0],  # Top-Left
    7:  [-0.29, -0.49, 0.0],  # Top-Right
    19: [ 0.29,  0.49, 0.0],  # Bottom-Left
    96: [-0.29,  0.49, 0.0]   # Bottom-Right
}

# ==========================================
# 3. ROS 2 NODE
# ==========================================

class DockingPublisher(Node):
    def __init__(self):
        super().__init__('docking_publisher')
        
        # --- Parameters ---
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('calibration_file', 'calibration_data.json')
        self.declare_parameter('marker_size', 0.15)
        self.declare_parameter('video_device', 0)
        self.declare_parameter('enable_gui', True) # Set False for Headless Robot
        self.declare_parameter('filter_alpha', 0.6) # 0.6 = moderate smoothing

        self.camera_frame = self.get_parameter('camera_frame').value
        self.calib_file = self.get_parameter('calibration_file').value
        self.marker_size = self.get_parameter('marker_size').value
        self.video_device = self.get_parameter('video_device').value
        self.enable_gui = self.get_parameter('enable_gui').value
        alpha = self.get_parameter('filter_alpha').value

        # --- Initialization ---
        self.mtx, self.dist = self.load_calibration(self.calib_file)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        
        # Configure Detector for Accuracy
        self.params = aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        # IPPE is best for planar markers (Requires OpenCV 4.x)
        if hasattr(aruco, 'SOLVEPNP_IPPE_SQUARE'):
            self.params.solvePnPMethod = aruco.SOLVEPNP_IPPE_SQUARE
        
        # Filter
        self.filter = PoseFilter(alpha_pos=alpha, alpha_rot=alpha)

        # Camera Setup
        self.cap = cv2.VideoCapture(self.video_device)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Failed to open device {self.video_device}")
            sys.exit(1)
            
        # --- LATENCY OPTIMIZATION ---
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # Buffer size 1 guarantees we always process the freshest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

        # QoS for fast updates (Volatile)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(PoseStamped, 'dock_pose', qos_profile)

        # Run slightly faster than 30Hz to clear buffer
        self.timer = self.create_timer(0.03, self.timer_callback)
        
        self.get_logger().info(f"Docking Optimized. GUI: {self.enable_gui}, Alpha: {alpha}")

    def load_calibration(self, path):
        if not os.path.exists(path):
            self.get_logger().fatal(f"Missing calibration: {path}")
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

    def get_center_from_marker(self, rvec, tvec, offset):
        R_mat, _ = cv2.Rodrigues(rvec)
        offset_world = np.dot(R_mat, np.array(offset))
        return tvec.flatten() + offset_world

    def timer_callback(self):
        # Latency Hack: Grab frame. If multiple frames are buffered, 
        # read() usually just pops the next one. 
        # To ensure freshness, we can grab() multiple times or rely on BUFFERSIZE=1.
        ret, frame = self.cap.read()
        
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is not None and len(ids) > 0:
            
            # Use detected markers to estimate pose
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.mtx, self.dist
            )

            # Find best marker (closest to center) and accumulate positions
            center_points = []
            best_rvec = None
            min_dist = float('inf')
            h, w = frame.shape[:2]
            img_center = np.array([w/2, h/2])

            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                if marker_id in DEFAULT_MARKER_MAP:
                    # Calculate world position
                    center_pt = self.get_center_from_marker(
                        rvecs[i], tvecs[i], DEFAULT_MARKER_MAP[marker_id]
                    )
                    center_points.append(center_pt)

                    # Orientation preference logic
                    marker_center = corners[i][0].mean(axis=0)
                    dist = np.linalg.norm(marker_center - img_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_rvec = rvecs[i]

            if center_points and best_rvec is not None:
                # 1. Raw Calculation
                raw_pos = np.mean(np.array(center_points), axis=0)
                
                rmat, _ = cv2.Rodrigues(best_rvec)
                raw_quat = R.from_matrix(rmat).as_quat() # x, y, z, w

                # 2. Filter (Smoothing)
                filt_pos, filt_quat = self.filter.update(raw_pos, raw_quat)

                # 3. Publish
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

                # 4. Visualization (Only if enabled)
                if self.enable_gui:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    # Draw coordinate axis at the SMOOTHED center
                    cv2.drawFrameAxes(frame, self.mtx, self.dist, best_rvec, filt_pos, 0.2)
                    
                    # Project center to screen
                    imgpts, _ = cv2.projectPoints(np.array([filt_pos]), np.zeros(3), np.zeros(3), self.mtx, self.dist)
                    c_x, c_y = imgpts[0].ravel().astype(int)
                    cv2.circle(frame, (c_x, c_y), 8, (0, 255, 0), -1) # Green dot = Filtered
                    
                    # Debug Info
                    text = f"XYZ: {filt_pos[0]:.2f}, {filt_pos[1]:.2f}, {filt_pos[2]:.2f}"
                    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if self.enable_gui:
            cv2.imshow('Optimized Docking', frame)
            if cv2.waitKey(1) == 27:
                rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DockingPublisher()
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

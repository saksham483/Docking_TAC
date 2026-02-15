#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from custom_msgs.msg import Telemetry 
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import os
import sys
import threading
from collections import deque
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ================= USER CONFIGURATION =================

# 1. HARDWARE
CAM_ID_FRONT = 0
CAM_ID_BOTTOM = 2
DEFAULT_CALIB_FILE = "calibration_data.json"

# 2. FILTERING (Your Parameters)
ALPHA_POS = 0.6      # Position Smoothing (0.6 = 60% New, 40% Old)
ALPHA_ROT = 0.6      # Rotation Smoothing
MAX_JUMP = 0.5       # Meters. Reject frames if jump > 0.5m
MEDIAN_BUFFER = 3    # Buffer size for median filtering

# 3. DOCKING BOARD SETUP (Your Map)
# Format: ID: [Offset_X, Offset_Y, Offset_Z] (Meters from Center)
MARKER_SIZE = 0.15   # Size of the black square (Meters)
BOARD_MAP = {
    28: [-0.29, -0.49, 0.0],  # Top-Left
    7:  [ 0.29, -0.49, 0.0],  # Top-Right
    19: [-0.29,  0.49, 0.0],  # Bot-Left
    96: [ 0.29,  0.49, 0.0]   # Bot-Right
}

# 4. STRATEGY
PHASE_1_SURGE_STOP = 0.85
PHASE_2_ALIGN_DIST = 0.15
HANDOVER_COAST_TIME = 2.5

# ================= 1. SMART FILTER (Your Logic) =================
class SmartPoseFilter:
    def __init__(self):
        self.alpha_pos = ALPHA_POS
        self.alpha_rot = ALPHA_ROT
        self.max_jump = MAX_JUMP
        self.prev_pos = None
        self.prev_quat = None
        self.pos_buffer = deque(maxlen=MEDIAN_BUFFER)
        self.consecutive_rejections = 0
        self.max_rejections = 5 

    def update(self, curr_pos, curr_quat):
        if self.prev_pos is None:
            self.prev_pos, self.prev_quat = curr_pos, curr_quat
            return curr_pos, curr_quat

        # Jump Rejection
        dist = np.linalg.norm(curr_pos - self.prev_pos)
        if dist > self.max_jump and self.consecutive_rejections < self.max_rejections:
            self.consecutive_rejections += 1
            return None, None
        
        self.consecutive_rejections = 0 # Reset if accepted
        
        # Median Stabilization
        self.pos_buffer.append(curr_pos)
        median_pos = np.median(np.array(self.pos_buffer), axis=0)

        # EMA Smoothing
        filt_pos = self.alpha_pos * median_pos + (1 - self.alpha_pos) * self.prev_pos

        # SLERP Smoothing
        try:
            rots = R.from_quat([self.prev_quat, curr_quat])
            slerp = Slerp([0, 1], rots)
            filt_quat = slerp([self.alpha_rot])[0].as_quat()
        except:
            filt_quat = curr_quat

        self.prev_pos, self.prev_quat = filt_pos, filt_quat
        return filt_pos, filt_quat

    def reset(self):
        self.prev_pos = None
        self.pos_buffer.clear()

# ================= 2. UTILS & THREADING =================
class AsyncCamera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            time.sleep(0.01)

    def get(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

def load_calibration(path):
    if not os.path.exists(path):
        print(f"ERROR: Calibration file {path} not found! Using defaults.")
        return np.array([[600,0,320],[0,600,240],[0,0,1]], dtype=float), np.zeros(5)
    with open(path, 'r') as f:
        data = json.load(f)
    return np.array(data['camera_matrix'], dtype=np.float32), \
           np.array(data['dist_coeff'], dtype=np.float32)

def generate_board_points():
    """Generates the 3D coordinates for all corners of the marker board."""
    pts = {}
    s = MARKER_SIZE / 2.0
    # Corners: TL, TR, BR, BL relative to marker center
    base_corners = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)
    
    for mid, offset in BOARD_MAP.items():
        # Add board offset to marker corners
        pts[mid] = base_corners + np.array(offset, dtype=np.float32)
    return pts

# ================= 3. MAIN NODE =================
class RobustDockingNode(Node):
    def __init__(self):
        super().__init__('robust_docking_node')
        
        # ROS Params & Topics
        self.declare_parameter('calib_file', DEFAULT_CALIB_FILE)
        calib_path = self.get_parameter('calib_file').value
        
        self.pub_error = self.create_publisher(Pose, '/control/pose_error', 10)
        self.pub_debug = self.create_publisher(Image, '/debug/perception', 10)
        self.bridge = CvBridge()
        
        # Load Calibration
        self.mtx, self.dist = load_calibration(calib_path)
        self.get_logger().info(f"Loaded Calibration: {calib_path}")

        # Vision Setup
        self.cam_front = AsyncCamera(CAM_ID_FRONT)
        self.cam_bottom = AsyncCamera(CAM_ID_BOTTOM)
        
        # Models & Detectors
        self.yolo = YOLO("best.pt")
        # Note: Using DICT_ARUCO_ORIGINAL as per your snippet
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        
        # Enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Logic Components
        self.board_pts_map = generate_board_points()
        self.filter_front = SmartPoseFilter()
        self.filter_bottom = SmartPoseFilter()
        
        self.state = "SEARCH"
        self.last_sight_time = time.time()
        
        # Start Loop
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Robust Docking Node: READY")

    def control_loop(self):
        frame_f = self.cam_front.get()
        frame_b = self.cam_bottom.get()
        if frame_f is None or frame_b is None: return

        # Enhance Bottom Cam (Underwater)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        gray_b = self.clahe.apply(gray_b)

        # --- PHASE 2: BOTTOM CAM (MULTI-MARKER BOARD) ---
        dock_pose = self.process_board(gray_b, frame_b)
        
        if dock_pose:
            self.state = "DOCK_LOCKED"
            self.last_sight_time = time.time()
            
            raw_p, raw_q = dock_pose
            final_p, final_q = self.filter_bottom.update(raw_p, raw_q)
            
            if final_p is not None:
                self.run_docking_logic(final_p, final_q)
                self.publish_debug(frame_f, frame_b, final_p)
                return

        # --- PHASE 1: FRONT CAM (YOLO) ---
        # If Bottom Cam lost/didn't find board, try Front Cam
        if self.state != "DOCK_LOCKED":
            appr_pose = self.process_yolo(frame_f)
            
            if appr_pose:
                self.state = "APPROACH"
                self.last_sight_time = time.time()
                
                raw_p, raw_q = appr_pose
                final_p, final_q = self.filter_front.update(raw_p, raw_q)
                
                if final_p is not None:
                    # Hold Depth (Z=0) during Phase 1
                    self.publish_cmd(final_p[0], final_p[1], 0.0, final_q)
                    self.publish_debug(frame_f, frame_b, final_p)
                    return

        # --- FAILSAFE: COAST or SEARCH ---
        elapsed = time.time() - self.last_sight_time
        if elapsed < HANDOVER_COAST_TIME:
            self.state = "COASTING"
            # Blindly push forward to bridge the camera gap
            self.publish_cmd(0.3, 0.0, 0.0, [0,0,0,1])
        else:
            self.state = "SEARCH"
            self.filter_front.reset()
            self.filter_bottom.reset()
            # Spin
            spin_q = R.from_euler('z', 15, degrees=True).as_quat()
            self.publish_cmd(0.0, 0.0, 0.0, spin_q)

        self.publish_debug(frame_f, frame_b, [0,0,0])

    # ================= ALGORITHMS =================
    
    def process_board(self, gray, debug_frame):
        """
        Robustly solves Pose using multiple markers. 
        Returns (pos, quat) of the BOARD CENTER.
        """
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None: return None
        
        obj_points = []
        img_points = []
        
        # Match detected IDs to our Board Map
        aruco.drawDetectedMarkers(debug_frame, corners, ids)
        
        ids_flat = ids.flatten()
        for i, mid in enumerate(ids_flat):
            if mid in self.board_pts_map:
                # Add the 4 corners of this marker from 3D Map
                obj_points.extend(self.board_pts_map[mid])
                # Add the 4 corners of this marker from 2D Image
                img_points.extend(corners[i][0])
        
        if len(obj_points) < 4: return None # Need at least 1 full marker
        
        # solvePnP on the aggregate cloud of points
        # This gives the pose of the CAMERA relative to the BOARD CENTER (0,0,0)
        success, rvec, tvec = cv2.solvePnP(
            np.array(obj_points), np.array(img_points), self.mtx, self.dist
        )
        
        if not success: return None
        
        cv2.drawFrameAxes(debug_frame, self.mtx, self.dist, rvec, tvec, 0.2)
        
        # --- COORDINATE TRANSFORM (Crucial) ---
        # tvec is Camera's position in Board Frame 
        # Actually solvePnP gives Object in Camera Frame.
        # Cam: X=Right, Y=Down, Z=Fwd
        # Body: X=Fwd, Y=Right, Z=Down
        
        cx, cy, cz = tvec.flatten()
        
        # Map to Body Errors
        # If board is at [0,0,5] in Cam (Z=5), Body Heave Error is 5.
        # If board is at [1,0,0] in Cam (X=1, Right), Body Sway Error is 1.
        # If board is at [0,-1,0] in Cam (Y=-1, Top), Body Surge Error is 1 (Forward).
        
        surge = -cy
        sway = cx
        heave = cz
        
        # Yaw
        # Simple heading alignment
        yaw = np.degrees(np.arctan2(sway, surge)) * 1.5
        q = R.from_euler('z', yaw, degrees=True).as_quat()
        
        return np.array([surge, sway, heave]), q

    def process_yolo(self, frame):
        """Standard Phase 1 Approach"""
        results = self.yolo(frame, verbose=False, stream=True)
        target = None
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    target = box.xywh[0].cpu().numpy()
                    break
        
        if target is None: return None
        
        cx, cy, w, h = target
        img_w = float(frame.shape[1])
        
        norm_x = (cx - (img_w/2)) / (img_w/2)
        width_ratio = w / img_w
        
        surge = 0.8 if width_ratio < PHASE_1_SURGE_STOP else 0.0
        sway = norm_x * 2.0
        yaw = norm_x * 35.0
        
        q = R.from_euler('z', yaw, degrees=True).as_quat()
        return np.array([surge, sway, 0.0]), q

    def run_docking_logic(self, pos, quat):
        surge, sway, heave = pos
        planar_err = np.sqrt(surge**2 + sway**2)
        
        z_cmd = 0.0
        if planar_err < PHASE_2_ALIGN_DIST:
            if heave > 0.30: # Stop height
                z_cmd = heave
            else:
                self.state = "DOCKED"
        
        self.publish_cmd(surge, sway, z_cmd, quat)

    def publish_cmd(self, x, y, z, q):
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.x = float(q[0])
        msg.orientation.y = float(q[1])
        msg.orientation.z = float(q[2])
        msg.orientation.w = float(q[3])
        self.pub_error.publish(msg)

    def publish_debug(self, front, bottom, cmd):
        if front.shape != bottom.shape:
            bottom = cv2.resize(bottom, (front.shape[1], front.shape[0]))
        combo = np.hstack((front, bottom))
        cv2.putText(combo, f"ST:{self.state} X:{cmd[0]:.2f} Y:{cmd[1]:.2f}", 
                   (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(combo, "bgr8"))
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = RobustDockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cam_front.running = False
        node.cam_bottom.running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from custom_msgs.msg import Telemetry  # Mira Stack Telemetry
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ================= CONFIGURATION =================
# --- Hardware Maps ---
CAM_ID_FRONT = 0
CAM_ID_BOTTOM = 2
CAM_WIDTH, CAM_HEIGHT = 640, 480

# --- Mira Topics ---
TOPIC_PID_ERROR = '/control/pose_error'  # Mira PID Listener
TOPIC_TELEMETRY = '/master/telemetry'    # Mira State Source
TOPIC_DEBUG_FRONT = '/debug/front_cam'
TOPIC_DEBUG_BOTTOM = '/debug/bottom_cam'

# --- Mission Parameters ---
TARGET_CLASS = 0
CONFIDENCE_THRESHOLD = 0.6
ARUCO_SIZE = 0.20 # Meters
PHASE_1_SURGE_TARGET = 0.85 # 85% width fill
PHASE_2_DESCEND_TOLERANCE = 0.15 # 15cm alignment required to dive

# --- AUV Physics Filters (The "SOTA" Part) ---
# Underwater vehicles lag. We need slow filters.
ALPHA_POS = 0.2      # Very smooth position (0.0=Frozen, 1.0=Raw)
ALPHA_ROT = 0.1      # Very smooth rotation
OUTLIER_DIST_M = 0.8 # Rejection gate (meters)
COAST_TIME = 3.0     # Seconds to drift blindly before searching

# --- Calibration (Replace with Mira's Calib) ---
K_MATRIX = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros(5)

# ================= HELPER CLASSES =================

class ImageEnhancer:
    """Enhances underwater images for better detection."""
    def __init__(self):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def apply(self, frame):
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L-channel (Luminance)
        cl = self.clahe.apply(l)
        # Merge and convert back
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

class RobustStateEstimator:
    """
    Handles Sensor Fusion & Outlier Rejection for AUVs.
    Uses LERP for Position, SLERP for Orientation.
    """
    def __init__(self):
        self.initialized = False
        self.pos = np.zeros(3) # x, y, z
        self.rot = R.from_quat([0, 0, 0, 1]) 
        self.outlier_count = 0
    
    def update(self, meas_pos, meas_yaw):
        meas_rot = R.from_euler('z', meas_yaw, degrees=True)
        
        if not self.initialized:
            self.pos = meas_pos
            self.rot = meas_rot
            self.initialized = True
            return self.pos, self.rot.as_euler('z', degrees=True)

        # 1. Outlier Rejection (Teleportation Check)
        dist = np.linalg.norm(meas_pos - self.pos)
        if dist > OUTLIER_DIST_M:
            self.outlier_count += 1
            if self.outlier_count < 5: # Wait 5 frames to confirm it's real
                return self.pos, self.rot.as_euler('z', degrees=True)
            self.outlier_count = 0
        
        # 2. LERP Position
        self.pos = (ALPHA_POS * meas_pos) + ((1 - ALPHA_POS) * self.pos)

        # 3. SLERP Rotation
        key_rots = R.from_quat([self.rot.as_quat(), meas_rot.as_quat()])
        slerp = Slerp([0, 1], key_rots)
        self.rot = slerp(ALPHA_ROT)

        return self.pos, self.rot.as_euler('z', degrees=True)

    def reset(self):
        self.initialized = False

class CameraThread:
    """Prevents blocking the ROS loop with IO operations"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            time.sleep(0.015)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.cap.release()

# ================= MAIN NODE =================

class MiraDockingPlanner(Node):
    def __init__(self):
        super().__init__('mira_docking_planner')
        
        # 1. Communications
        self.pub_error = self.create_publisher(Pose, TOPIC_PID_ERROR, 10)
        self.pub_dbg_front = self.create_publisher(Image, TOPIC_DEBUG_FRONT, 10)
        self.pub_dbg_bottom = self.create_publisher(Image, TOPIC_DEBUG_BOTTOM, 10)
        
        self.sub_telem = self.create_subscription(
            Telemetry, TOPIC_TELEMETRY, self.telemetry_cb, 10
        )
        self.bridge = CvBridge()

        # 2. Modules
        self.estimator = RobustStateEstimator()
        self.enhancer = ImageEnhancer()
        
        try:
            # Enable tracking for ID persistence
            self.yolo = YOLO("best.pt") 
        except:
            self.get_logger().error("YOLO 'best.pt' not found!")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 3. State
        self.state = "SEARCH"
        self.last_sight_time = time.time()
        self.current_depth = 0.0
        self.current_yaw = 0.0

        # 4. Hardware
        self.cam_front = CameraThread(CAM_ID_FRONT)
        self.cam_bottom = CameraThread(CAM_ID_BOTTOM)

        # 5. Timer
        self.create_timer(0.05, self.control_loop) # 20Hz Control Loop
        self.get_logger().info("Mira Docking Node: OPERATIONAL")

    def telemetry_cb(self, msg):
        self.current_depth = msg.depth
        self.current_yaw = msg.yaw

    def control_loop(self):
        # 1. Acquire & Enhance Frames
        raw_f = self.cam_front.read()
        raw_b = self.cam_bottom.read()
        
        if raw_f is None or raw_b is None: return

        frame_f = self.enhancer.apply(raw_f) # Apply Underwater filter
        frame_b = self.enhancer.apply(raw_b)

        # 2. Logic Pipeline (Priority: Bottom -> Front -> Coast -> Search)
        
        # --- CHECK BOTTOM CAM (PHASE 2) ---
        aruco_pose = self.process_aruco(frame_b)
        
        if aruco_pose is not None:
            self.state = "PHASE_2_LOCK"
            self.last_sight_time = time.time()
            
            # Smooth the noisy raw pose
            smooth_pos, smooth_yaw = self.estimator.update(aruco_pose[:3], aruco_pose[3])
            
            # Execute descent logic
            self.execute_docking(smooth_pos, smooth_yaw)
            self.publish_debug(frame_f, frame_b)
            return

        # --- CHECK FRONT CAM (PHASE 1) ---
        yolo_target = self.process_yolo(frame_f)
        
        if yolo_target is not None:
            self.state = "PHASE_1_APPROACH"
            self.last_sight_time = time.time()
            
            # Convert YOLO bbox to "Pseudo-Pose" for the estimator
            raw_pos, raw_yaw = self.yolo_to_pose(yolo_target, frame_f.shape)
            
            # Smooth it
            smooth_pos, smooth_yaw = self.estimator.update(raw_pos, raw_yaw)
            
            self.publish_pid(smooth_pos[0], smooth_pos[1], 0.0, smooth_yaw) # Z=0 means hold depth
            self.publish_debug(frame_f, frame_b)
            return

        # --- LOST TARGET LOGIC ---
        elapsed = time.time() - self.last_sight_time
        
        if elapsed < COAST_TIME:
            # Coast Mode: Maintain last known vector (AUV Momentum)
            # This bridges the gap between Front Cam loss and Bottom Cam acquisition
            self.get_logger().info(f"Coasting... {elapsed:.1f}s")
            # Send small forward surge to ensure we cross over the marker
            self.publish_pid(0.3, 0.0, 0.0, 0.0) 
        else:
            # Search Mode: Spin
            self.state = "SEARCH"
            self.estimator.reset()
            self.publish_pid(0.0, 0.0, 0.0, 10.0) # 10 deg/s yaw

        self.publish_debug(frame_f, frame_b)

    def process_aruco(self, frame):
        """Returns [Surge, Sway, Depth, Yaw] or None"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None: return None

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], ARUCO_SIZE, K_MATRIX, DIST_COEFFS)
        cv2.drawFrameAxes(frame, K_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)

        # --- NED TRANSFORM FOR MIRA ---
        # Cam Frame: X=Right, Y=Down, Z=Forward
        # Mira Body: X=Surge(Fwd), Y=Sway(Right), Z=Heave(Down)
        
        # Bottom Cam (Looking Down):
        # Cam Z = Depth Distance (Heave Error)
        # Cam X = Body Sway (Right) -> +X in Cam is +Y in Body
        # Cam Y = Body Surge (Backwards) -> +Y in Cam is -X in Body
        
        tx, ty, tz = tvec[0][0]
        
        surge_err = -ty 
        sway_err = tx   
        heave_err = tz  
        
        # Yaw: Simple Heading alignment
        yaw_err = np.degrees(np.arctan2(tx, ty)) * 1.5

        return np.array([surge_err, sway_err, heave_err, yaw_err])

    def process_yolo(self, frame):
        # Use track=True for temporal consistency
        results = self.yolo.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD)
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == TARGET_CLASS:
                    # Draw for debug
                    x,y,w,h = box.xywh[0].cpu().numpy()
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0), 2)
                    return box.xywh[0].cpu().numpy()
        return None

    def yolo_to_pose(self, bbox, shape):
        cx, cy, w, h = bbox
        img_h, img_w, _ = shape

        # Normalize Errors
        # X Error (Sway): Center of box vs Center of Image
        # -1 (Left) to +1 (Right)
        norm_x = (cx - (img_w/2)) / (img_w/2)
        
        # Surge: Based on size. If small, surge 1.0. If big, surge 0.0.
        size_ratio = w / img_w
        surge_cmd = 1.0 if size_ratio < PHASE_1_SURGE_TARGET else 0.0

        # Create specific gains for AUV
        sway_cmd = norm_x * 2.0 
        yaw_cmd = norm_x * 35.0 # Turn into the sway
        
        return np.array([surge_cmd, sway_cmd, 0.0]), yaw_cmd

    def execute_docking(self, pos, yaw):
        surge, sway, depth = pos
        
        planar_err = np.sqrt(surge**2 + sway**2)
        
        z_cmd = 0.0
        
        # Safety: Only descend if we are stable over the marker
        if planar_err < PHASE_2_DESCEND_TOLERANCE:
            if depth > 0.35: # Don't crash into the floor
                z_cmd = depth # Request PID to go down by this amount
            else:
                self.state = "DOCKED"
                z_cmd = 0.0
        
        self.publish_pid(surge, sway, z_cmd, yaw)

    def publish_pid(self, x, y, z, yaw_deg):
        """Publishes to Mira's control topic"""
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        
        q = R.from_euler('z', yaw_deg, degrees=True).as_quat()
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]
        
        self.pub_error.publish(msg)

    def publish_debug(self, frame_f, frame_b):
        cv2.putText(frame_f, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        try:
            self.pub_dbg_front.publish(self.bridge.cv2_to_imgmsg(frame_f, "bgr8"))
            self.pub_dbg_bottom.publish(self.bridge.cv2_to_imgmsg(frame_b, "bgr8"))
        except: pass

    def destroy_node(self):
        self.cam_front.release()
        self.cam_bottom.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MiraDockingPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

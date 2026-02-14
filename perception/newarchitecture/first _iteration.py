#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from custom_msgs.msg import Telemetry 
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
# Hardware
CAM_ID_FRONT = 0
CAM_ID_BOTTOM = 2

# Topics
TOPIC_PID_ERROR = '/control/pose_error'  # PID Node listens to this
TOPIC_TELEMETRY = '/master/telemetry'    # To read Current Depth/Yaw

# Strategy
MODEL_PATH = "best.pt"
TARGET_CLASS = 0
PHASE_1_SURGE_TARGET = 0.70  # Target Width Ratio (Front Cam)
PHASE_2_DESCEND_GAP = 0.20   # Meters (Alignment threshold)
PHASE_2_STOP_HEIGHT = 0.30   # Meters (Final Docking Height)

# Calibration (Bottom Cam)
FX, FY, CX, CY = 600, 600, 320, 240 # need to change it according to calibration matrix
DIST_COEFFS = np.zeros(5)
ARUCO_SIZE = 0.20 # Meters
# =================================================

class CameraThread:
    """Efficiently reads frames in background thread"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.cap.release()

class DockingBrain(Node):
    def __init__(self):
        super().__init__('docking_brain')

        # 1. State Variables
        self.current_depth = 0.0
        self.current_yaw = 0.0
        self.state = "SEARCH"
        self.last_sight_time = 0.0

        # 2. Hardware & AI
        self.get_logger().info("Starting Cameras & YOLO...")
        self.cam_front = CameraThread(CAM_ID_FRONT)
        self.cam_bottom = CameraThread(CAM_ID_BOTTOM)
        
        try:
            self.yolo = YOLO(MODEL_PATH)
        except:
            self.get_logger().error("YOLO Model not found! Check path.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 3. ROS Communications
        self.pub_error = self.create_publisher(Pose, TOPIC_PID_ERROR, 10)
        
        # Subscribe to Custom Telemetry for Depth/Heading Hold
        self.sub_telem = self.create_subscription(
            Telemetry, TOPIC_TELEMETRY, self.telemetry_cb, 10
        )

        self.create_timer(0.05, self.control_loop) # 20Hz
        self.get_logger().info("Docking Brain Ready.")

    def telemetry_cb(self, msg):
        # We need this to override Z setpoint to "Current Depth"
        self.current_depth = msg.depth
        self.current_yaw = msg.yaw

    def publish_error(self, x_surge, y_sway, z_err, yaw_err_deg):
        """
        Publishes Body-Frame Error Vector.
        PID Logic: Output = Kp * Error.
        """
        msg = Pose()
        msg.position.x = float(x_surge)
        msg.position.y = float(y_sway)
        msg.position.z = float(z_err) # Vertical Distance to go

        # Convert Yaw Deg -> Quaternion
        q = R.from_euler('z', yaw_err_deg, degrees=True).as_quat()
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        self.pub_error.publish(msg)

    def control_loop(self):
        frame_front = self.cam_front.read()
        frame_bottom = self.cam_bottom.read()

        if frame_front is None or frame_bottom is None: return

        # --- PHASE 2: BOTTOM CAM (Priority) ---
        gray = cv2.cvtColor(frame_bottom, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            self.run_phase_2(corners, ids)
        else:
            # --- PHASE 1: FRONT CAM (YOLO) ---
            self.run_phase_1(frame_front)

    def run_phase_1(self, frame):
        # YOLO Detection
        results = self.yolo(frame, verbose=False, stream=True)
        target = None
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == TARGET_CLASS:
                    target = box.xywh[0].cpu().numpy()
                    break
            if target: break

        if target is not None:
            self.state = "PHASE_1"
            self.last_sight_time = time.time()
            cx, cy, w, h = target
            img_h, img_w, _ = frame.shape

            # 1. SURGE: Approach until box fills 85% width
            # Error > 0 means "Move Forward"
            surge_err = ((img_w * PHASE_1_SURGE_TARGET) - w) / img_w * 4.0
            
            # 2. SWAY: Center X
            sway_err = (cx - (img_w/2)) / img_w * 2.0
            
            # 3. YAW: Turn towards Sway
            yaw_err = sway_err * 30.0

            # 4. HEAVE: Override Z to Current Depth (Hold Level)
            # We send 0.0 Z-error. PID maintains current depth.
            z_err = 0.0 

            self.publish_error(surge_err, sway_err, z_err, yaw_err)
        
        else:
            self.handle_blind_spot()

    def run_phase_2(self, corners, ids):
        self.state = "PHASE_2"
        self.last_sight_time = time.time()

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, ARUCO_SIZE, 
            np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]]), DIST_COEFFS
        )
        tvec = tvecs[0][0] # x, y, z in Camera Frame

        # COORDINATE TRANSFORM (Bottom Cam -> Body)
        # Cam X (Right) -> Body Y (Sway)
        # Cam Y (Down in image) -> Body -X (Surge/Forward)
        # Cam Z (Depth) -> Body Z (Depth)
        
        surge_err = -tvec[1]
        sway_err = tvec[0]
        dist_down = tvec[2]

        yaw_err = sway_err * 45.0 # Simple alignment

        # DESCENT LOGIC
        # Only descend if we are aligned (XY error is small)
        planar_err = np.sqrt(surge_err**2 + sway_err**2)

        if planar_err < PHASE_2_DESCEND_GAP:
            if dist_down > PHASE_2_STOP_HEIGHT:
                # Aligned & High: Go Down.
                # Send Distance as Error. PID will thrust down.
                z_err = dist_down
            else:
                # Aligned & Low: Stop.
                z_err = 0.0
        else:
            # Not Aligned: Hover at Current Depth
            z_err = 0.0

        self.publish_error(surge_err, sway_err, z_err, yaw_err)

    def handle_blind_spot(self):
        # If lost recently (<3s), push forward blind
        if self.state == "PHASE_1" and (time.time() - self.last_sight_time) < 3.0:
            # Surge 1.0m, Hold Depth (0.0), Hold Heading (0.0 yaw error)
            self.publish_error(1.0, 0.0, 0.0, 0.0)
        else:
            # Search Spin
            self.state = "SEARCH"
            self.publish_error(0.0, 0.0, 0.0, 15.0)

    def destroy_node(self):
        self.cam_front.release()
        self.cam_bottom.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DockingBrain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from custom_msgs.msg import Telemetry 
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# ================= CONFIGURATION =================
# --- Hardware ---
CAM_ID_FRONT = 0
CAM_ID_BOTTOM = 2
CAM_WIDTH, CAM_HEIGHT = 640, 480

# --- Topics ---
TOPIC_PID_ERROR = '/control/pose_error'  # Output to PID
TOPIC_TELEMETRY = '/master/telemetry'    # Input Current State
TOPIC_DEBUG_FRONT = '/debug/front_cam'
TOPIC_DEBUG_BOTTOM = '/debug/bottom_cam'

# --- Strategy Parameters ---
TARGET_CLASS = 0                # YOLO Class ID
PHASE_1_SURGE_THRESHOLD = 0.80  # Switch to Phase 2 when box width > 80% of screen
PHASE_2_ALIGN_TOLERANCE = 0.15  # Meters (XY error must be < this to descend)
PHASE_2_STOP_HEIGHT = 0.30      # Meters (Stop descending here)
LOSS_PATIENCE = 2.0             # Seconds to wait before switching back to SEARCH

# --- Calibration (Intrinsics) ---
# REPLACE THESE WITH REAL CALIBRATION VALUES FOR ACCURACY
K_MATRIX = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.zeros(5)
ARUCO_SIZE = 0.20  # Meters

# ================= HELPER CLASSES =================

class MovingAverageFilter:
    """Smooths out jittery detection data."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None

class CameraThread:
    """Reads frames in background to prevent buffer lag."""
    def __init__(self, src, name):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.name = name
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            time.sleep(0.01) # Small sleep to prevent CPU hogging

    def read(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.cap.release()

# ================= MAIN NODE =================

class DockingPlanner(Node):
    def __init__(self):
        super().__init__('docking_planner')
        
        # 1. Communications
        self.pub_error = self.create_publisher(Pose, TOPIC_PID_ERROR, 10)
        self.pub_dbg_front = self.create_publisher(Image, TOPIC_DEBUG_FRONT, 10)
        self.pub_dbg_bottom = self.create_publisher(Image, TOPIC_DEBUG_BOTTOM, 10)
        
        self.sub_telem = self.create_subscription(
            Telemetry, TOPIC_TELEMETRY, self.telemetry_cb, 10
        )
        self.bridge = CvBridge()

        # 2. State & Telemetry
        self.current_depth = 0.0
        self.current_yaw = 0.0
        self.state = "SEARCH" # SEARCH, APPROACH, ALIGN, DOCKING, DONE
        self.last_detection_time = time.time()
        
        # 3. Vision Models
        self.get_logger().info(f"Loading YOLO Model...")
        try:
            self.yolo = YOLO("best.pt") # Ensure this file is in local dir
        except Exception as e:
            self.get_logger().error(f"YOLO Load Failed: {e}")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # 4. Filters (Smooth Control)
        self.filter_x = MovingAverageFilter(0.7)
        self.filter_y = MovingAverageFilter(0.7)
        self.filter_yaw = MovingAverageFilter(0.5)

        # 5. Hardware Init
        self.cam_front = CameraThread(CAM_ID_FRONT, "Front")
        self.cam_bottom = CameraThread(CAM_ID_BOTTOM, "Bottom")
        
        # 6. Loop
        self.timer = self.create_timer(0.05, self.control_loop) # 20 Hz
        self.get_logger().info("Docking Planner Initialized.")

    def telemetry_cb(self, msg):
        self.current_depth = msg.depth
        self.current_yaw = msg.yaw

    def publish_control(self, x_err, y_err, z_setpoint_mode, z_val, yaw_err):
        """
        x_err, y_err: Meters (or normalized ratio) relative to body.
        z_setpoint_mode: 'ERR' (velocity/distance) or 'ABS' (depth hold).
        z_val: The value for Z.
        yaw_err: Degrees relative to body.
        """
        msg = Pose()
        
        # Position Errors
        msg.position.x = float(x_err)
        msg.position.y = float(y_err)
        
        # Z Logic: If we are docking, we send distance to go. 
        # If searching, we send 0.0 error but might need specific depth handling logic in PID.
        # Here we assume PID takes Z as Error.
        if z_setpoint_mode == 'ABS':
            # If PID expects Error, and we want to hold depth:
            # Error = Desired - Current. 
            # If we want to hold current depth, Error = 0.
            msg.position.z = 0.0 
        else:
            msg.position.z = float(z_val) 

        # Orientation (Yaw Error -> Quaternion)
        # We only care about Yaw for alignment. Pitch/Roll are stabilized.
        q = R.from_euler('z', yaw_err, degrees=True).as_quat()
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        self.pub_error.publish(msg)

    def control_loop(self):
        # 1. Get Frames
        frame_front = self.cam_front.read()
        frame_bottom = self.cam_bottom.read()

        if frame_front is None or frame_bottom is None:
            return

        # 2. Check Bottom Cam (Highest Priority)
        # If we see the marker, we ignore the front camera (Phase 2)
        bottom_pose = self.process_aruco(frame_bottom)
        
        if bottom_pose is not None:
            self.execute_phase_2_docking(bottom_pose)
            self.last_detection_time = time.time()
            self.publish_debug(frame_front, frame_bottom)
            return

        # 3. If no ArUco, Check Front Cam (Phase 1)
        front_target = self.process_yolo(frame_front)
        
        if front_target is not None:
            self.execute_phase_1_approach(front_target, frame_front.shape)
            self.last_detection_time = time.time()
        else:
            # 4. Blind Spot / Search Logic
            self.handle_search_or_blind()

        self.publish_debug(frame_front, frame_bottom)

    # ================= PERCEPTION PIPELINE =================

    def process_yolo(self, frame):
        """Returns (cx, cy, w, h) of target or None"""
        results = self.yolo(frame, verbose=False, stream=True, conf=0.5)
        
        best_box = None
        max_area = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == TARGET_CLASS:
                    xywh = box.xywh[0].cpu().numpy()
                    area = xywh[2] * xywh[3]
                    if area > max_area:
                        max_area = area
                        best_box = xywh
        
        # Draw on frame for debug
        if best_box is not None:
            x, y, w, h = best_box
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
            
        return best_box

    def process_aruco(self, frame):
        """
        Returns (surge_err, sway_err, dist_z, yaw_err) in BODY FRAME or None.
        Uses solvePnP for accurate 3D pose.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return None

        # Draw markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate Pose (PnP)
        # Assuming marker is flat on the ground
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], ARUCO_SIZE, K_MATRIX, DIST_COEFFS)
        
        # Draw axis
        cv2.drawFrameAxes(frame, K_MATRIX, DIST_COEFFS, rvec, tvec, 0.1)

        # --- COORDINATE TRANSFORM (Crucial) ---
        # Camera Frame: X=Right, Y=Down, Z=Forward (Depth)
        # Body Frame: X=Forward, Y=Left, Z=Up
        #
        # Setup: Bottom camera looking DOWN.
        # Cam X (Right) -> Body Y (Left? No, Right Sway is -Y usually, let's assume Body Y is Left)
        # Cam Y (Down in image) -> Body -X (Backward in Body) -> so -CamY = Body X
        # Cam Z (Depth) -> Body -Z (Down)
        
        # Raw Tvec (Camera Frame)
        x_c, y_c, z_c = tvec[0][0], tvec[0][1], tvec[0][2]

        # Body Frame Conversion
        # Surge (Body X): Corresponds to negative Image Y
        # If target is at top of image (small y), it is forward (positive X)
        surge_err = -y_c 
        
        # Sway (Body Y): Corresponds to Image X
        # If target is Right (positive x), Body needs to move Right.
        # If Body Y is +Left, then we need -Y.
        sway_err = -x_c 
        
        # Vertical Distance
        dist_down = z_c

        # Yaw Calculation from Rotation Matrix
        rmat, _ = cv2.Rodrigues(rvec)
        # Calculate angle between Camera X-axis and Marker X-axis
        # This part requires tuning based on marker placement. 
        # Simplification: Use atan2 of tvec for approach, but for alignment, we need R matrix.
        # Let's approximate Yaw error using x/y offset for now (Head-to-target) 
        # OR extracting euler angles.
        
        # Simple Heading approach (point nose at marker)
        yaw_err = np.degrees(np.arctan2(sway_err, surge_err)) * 2.0 # Gain P
        
        # Refined Orientation Alignment (Aligning axes)
        # Extract yaw from rotation matrix relative to camera
        # ... (Advanced math omitted for robustness, using Position alignment first)

        return (surge_err, sway_err, dist_down, yaw_err)

    # ================= CONTROL STRATEGIES =================

    def execute_phase_1_approach(self, target_box, shape):
        self.state = "APPROACH"
        cx, cy, w, h = target_box
        img_h, img_w, _ = shape

        # Normalize Errors (-1 to 1)
        # Center of image is target
        err_x = (cx - (img_w / 2)) / (img_w / 2) # Sway
        # Target width ratio
        width_ratio = w / img_w

        # Control Logic
        # Sway: Proportional to X error
        sway_cmd = -err_x * 2.0 # Gain

        # Yaw: Turn into the sway to assist
        yaw_cmd = -err_x * 30.0 # Degrees

        # Surge: Move forward until box is big enough
        if width_ratio < PHASE_1_SURGE_THRESHOLD:
            surge_cmd = 1.0 # Move forward at constant speed
        else:
            surge_cmd = 0.0 # Stop, we are close (handover to bottom cam expected)

        # Filters
        sway_cmd = self.filter_y.update(sway_cmd)
        yaw_cmd = self.filter_yaw.update(yaw_cmd)

        self.get_logger().info(f"P1: Surge {surge_cmd:.2f}, Size {width_ratio:.2f}")
        self.publish_control(surge_cmd, sway_cmd, 'ABS', 0.0, yaw_cmd)

    def execute_phase_2_docking(self, pose):
        self.state = "ALIGN/DOCK"
        surge_raw, sway_raw, dist_down, yaw_raw = pose

        # Apply Filters
        surge_err = self.filter_x.update(surge_raw)
        sway_err = self.filter_y.update(sway_raw)
        yaw_err = self.filter_yaw.update(yaw_raw)

        # Planar Error (How well centered are we?)
        planar_dist = np.sqrt(surge_err**2 + sway_err**2)

        z_cmd = 0.0
        
        # Logic: Align first, then Descend
        if planar_dist < PHASE_2_ALIGN_TOLERANCE:
            if dist_down > PHASE_2_STOP_HEIGHT:
                # We are aligned, go down
                # PID expects positive Z error to go down? Or negative?
                # Assuming PID: Thrust = Kp * Error. To go down, we need to aim for deeper depth.
                # Here we pass 'ERR' mode. Distance to target.
                z_cmd = dist_down 
                self.get_logger().info("P2: DESCENDING")
            else:
                # We are at bottom
                z_cmd = 0.0
                self.state = "DOCKED"
                self.get_logger().info("P2: DOCKED")
        else:
            # High error, hover and align
            z_cmd = 0.0
            self.get_logger().info(f"P2: ALIGNING (Err: {planar_dist:.2f}m)")

        self.publish_control(surge_err, sway_err, 'ERR', z_cmd, yaw_err)

    def handle_search_or_blind(self):
        elapsed = time.time() - self.last_detection_time
        
        if elapsed < LOSS_PATIENCE:
            # Blind Spot Handling: Keep moving forward blindly for a bit
            # This helps if the object momentarily leaves the camera frame (e.g. "too close")
            self.get_logger().warn(f"Blind Spot: Holding course... ({elapsed:.1f}s)")
            # Surge 0.5m, Hold other axes
            self.publish_control(0.5, 0.0, 'ABS', 0.0, 0.0)
        else:
            # Search Mode
            self.state = "SEARCH"
            self.filter_x.reset()
            self.filter_y.reset()
            
            # Spin slowly to find target
            self.get_logger().info("Searching...")
            self.publish_control(0.0, 0.0, 'ABS', 0.0, 15.0) # 15 deg/s Yaw

    def publish_debug(self, frame_front, frame_bottom):
        # Annotate State
        cv2.putText(frame_front, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        try:
            msg_front = self.bridge.cv2_to_imgmsg(frame_front, "bgr8")
            msg_bottom = self.bridge.cv2_to_imgmsg(frame_bottom, "bgr8")
            self.pub_dbg_front.publish(msg_front)
            self.pub_dbg_bottom.publish(msg_bottom)
        except Exception:
            pass

    def destroy_node(self):
        self.cam_front.release()
        self.cam_bottom.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DockingPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

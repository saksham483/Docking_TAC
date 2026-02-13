#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
import sys
from scipy.spatial.transform import Rotation as R, Slerp

class PoseFilter:
    def __init__(self, alpha_pos=0.7, alpha_rot=0.7):
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.prev_pos = None
        self.prev_quat = None

    def update(self, curr_pos, curr_quat):
        if self.prev_pos is None:
            self.prev_pos = curr_pos
            self.prev_quat = curr_quat
            return curr_pos, curr_quat
        filt_pos = self.alpha_pos * curr_pos + (1 - self.alpha_pos) * self.prev_pos
        key_times = [0, 1]
        key_rots = R.from_quat([self.prev_quat, curr_quat])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([self.alpha_rot]) 
        filt_quat = interp_rot[0].as_quat()
        self.prev_pos = filt_pos
        self.prev_quat = filt_quat
        return filt_pos, filt_quat

DEFAULT_MARKER_MAP = {
    28: [ 0.29, -0.49, 0.0],
    7:  [-0.29, -0.49, 0.0],
    19: [ 0.29,  0.49, 0.0],
    96: [-0.29,  0.49, 0.0]
}

class DockingPublisher(Node):
    def __init__(self):
        super().__init__('docking_publisher')
        
        # --- Parameters ---
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('calibration_file', 'calibration_data.json')
        self.declare_parameter('marker_size', 0.15)
        self.declare_parameter('video_device', 0)
        self.declare_parameter('enable_gui', True)
        self.declare_parameter('filter_alpha', 0.6)
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('center_deadzone', 0.1)

        self.camera_frame = self.get_parameter('camera_frame').value
        self.calib_file = self.get_parameter('calibration_file').value
        self.marker_size = self.get_parameter('marker_size').value
        self.video_device = self.get_parameter('video_device').value
        self.enable_gui = self.get_parameter('enable_gui').value
        alpha = self.get_parameter('filter_alpha').value
        self.stop_dist = self.get_parameter('stop_distance').value
        self.deadzone = self.get_parameter('center_deadzone').value

        # --- Initialization ---
        self.mtx, self.dist = self.load_calibration(self.calib_file)
        
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.params = aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        
        self.filter = PoseFilter(alpha_pos=alpha, alpha_rot=alpha)

        self.cap = cv2.VideoCapture(self.video_device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Failed to open device {self.video_device}")
            sys.exit(1)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.publisher_ = self.create_publisher(PoseStamped, 'dock_pose', qos_profile)
        self.dir_publisher_ = self.create_publisher(String, 'move_direction', 10)
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Docking Node Started. Logging Euler angles to terminal.")

    def load_calibration(self, path):
        if not os.path.exists(path):
            self.get_logger().fatal(f"Missing calibration: {path}")
            sys.exit(1)
        with open(path, 'r') as f:
            data = json.load(f)
        return np.array(data['camera_matrix'], dtype=np.float32), \
               np.array(data['dist_coeff'], dtype=np.float32)

    def get_center_from_marker(self, rvec, tvec, offset):
        R_mat, _ = cv2.Rodrigues(rvec)
        offset_world = np.dot(R_mat, np.array(offset))
        return tvec.flatten() + offset_world

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        
        cmd = "SEARCHING"

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.mtx, self.dist
            )

            center_points = []
            best_rvec = None
            min_dist = float('inf')
            h, w = frame.shape[:2]
            img_center = np.array([w/2, h/2])

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in DEFAULT_MARKER_MAP:
                    center_pt = self.get_center_from_marker(rvecs[i], tvecs[i], DEFAULT_MARKER_MAP[marker_id])
                    center_points.append(center_pt)
                    marker_center = corners[i][0].mean(axis=0)
                    dist = np.linalg.norm(marker_center - img_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_rvec = rvecs[i]

            if center_points and best_rvec is not None:
                raw_pos = np.mean(np.array(center_points), axis=0)
                rmat, _ = cv2.Rodrigues(best_rvec)
                raw_quat = R.from_matrix(rmat).as_quat()

                filt_pos, filt_quat = self.filter.update(raw_pos, raw_quat)

                # --- Euler Angle Logging ---
                rot_obj = R.from_quat(filt_quat)
                roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=True)
                
                self.get_logger().info(
                    f"Orientation (XYZ Deg) -> R: {roll:.1f}, P: {pitch:.1f}, Y: {yaw:.1f}"
                )

                # Direction Logic
                x_pos = filt_pos[0]
                z_dist = filt_pos[2]
                
                if x_pos > self.deadzone: cmd = "GO RIGHT"
                elif x_pos < -self.deadzone: cmd = "GO LEFT"
                elif z_dist > self.stop_dist: cmd = "GO FORWARD"
                else: cmd = "STOP / ARRIVED"
                
                # Publish Pose
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_frame
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, filt_pos)
                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, filt_quat)
                self.publisher_.publish(msg)

                if self.enable_gui:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.mtx, self.dist, best_rvec, filt_pos, 0.2)
                    cv2.putText(frame, f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Publish direction text
        self.dir_publisher_.publish(String(data=cmd))

        # Local GUI display
        if self.enable_gui:
            cv2.putText(frame, f"CMD: {cmd}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Docking Monitor', frame)
            if cv2.waitKey(1) == 27:
                rclpy.shutdown()

    def destroy_node(self):
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
        rclpy.shutdown()

if __name__ == "__main__":
    main()

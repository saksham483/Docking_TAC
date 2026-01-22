# Docking_TAC
Docking part of the TAC challenge 2026


rosref.py is the code with the filter and latency optimization 
Implements an Exponential Moving Average (Low Pass Filter) to smooth out jittery ArUco detections.
filter position using linear interpolation
filter rotation using the spherical linear interpolation 




whereas the puck.py is only a implementation of board create method without any ros2 publisher and filtes


points to remember before executing the rosref.py
-> there should be a json file with the calibration data named as calibratin_data.json
-> docking map size should be cross checke with the existing markermap  
DEFAULT_MARKER_MAP = {
    28: [ 0.29, -0.49, 0.0],  # Top-Left
    7:  [-0.29, -0.49, 0.0],  # Top-Right
    19: [ 0.29,  0.49, 0.0],  # Bottom-Left
    96: [-0.29,  0.49, 0.0]   # Bottom-Right
}
here 0.29 and 0.49 are in meters 
-> opencv 4.x + is required for this script

**Controls pipeline**

added a controller pipeline dock_controller.py 
some things needs attention before executing the script
-> path of the publisher dock_pose needs to be verified for the subscriber in the script 
->  coordinate frame to be aligned according to the camera orientation as there could be a difference in the axis
-> blind spot distance is hardcoded as 0.5 meters change it according to the camera's FOV
-> the SEARCH state involves surging forward so make sure to keep bot away from the wall in front 
-> kp_sway and kp_yaw are to be changed according to the behaviour

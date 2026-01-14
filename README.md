# Docking_TAC
Docking part of the TAC challenge 2026


rosref.py is the code with the filter and latency optimization 
Implements an Exponential Moving Average (Low Pass Filter) to smooth out jittery ArUco detections.
filter position using linear interpolation
filter rotation using the spherical linear interpolation 




whereas the puck.py is only a implementation of board create method without any ros2 publisher and filtes

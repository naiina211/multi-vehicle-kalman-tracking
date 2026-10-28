# Multi-Vehicle Tracking Using Kalman Filter

Tracks vehicles in any input video using YOLOv8 detection and a
per-vehicle Kalman Filter (constant-velocity model) with Hungarian
association.

## Install
pip install opencv-python numpy scipy ultralytics

## Run
python vehicle_tracker.py --input traffic.mp4 --output result.mp4 --show

## Options
--detector yolo|motion, --max-missed, --min-hits, --max-dist, --conf

## Output Video
[Watch the tracking result]https://drive.google.com/file/d/15q2S4S1XKLbZP9Mf791-mJMhdg-ptvZS/view?usp=drive_link

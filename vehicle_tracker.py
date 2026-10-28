import argparse
import sys
from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}  # COCO ids

# Detectors
class YoloDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        res = self.model(frame, conf=self.conf,
                         classes=list(VEHICLE_CLASSES.keys()), verbose=False)[0]
        dets = []
        for box, c in zip(res.boxes.xyxy.cpu().numpy(),
                          res.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box
            dets.append((x1, y1, x2, y2, VEHICLE_CLASSES.get(c, "Vehicle")))
        return dets


class MotionDetector:
    """Fallback detector: no model download needed (works for static camera)."""

    def __init__(self, min_area=800):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300,
                                                     varThreshold=40,
                                                     detectShadows=True)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        mask = self.bg.apply(frame)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                dets.append((x, y, x + w, y + h, "Vehicle"))
        return dets

# Kalman-filter track
class Track:
    _next_id = 1

    def __init__(self, det):
        x1, y1, x2, y2, cls = det
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.w, self.h, self.cls = x2 - x1, y2 - y1, cls
        self.id = Track._next_id
        Track._next_id += 1

        # State = [x, y, vx, vy]  (constant-velocity model)
        # Measurement = [x, y]    (detected centroid)
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([1, 1, 5, 5]).astype(np.float32)  # Q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4       # R
        kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)  # P0
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf = kf

        self.pred = (cx, cy)
        self.hits = 1            # total detections matched
        self.missed = 0          # consecutive frames without detection
        self.detected_now = True
        self.trail = deque(maxlen=30)
        self.trail.append((int(cx), int(cy)))

    def predict(self):
        p = self.kf.predict()
        self.pred = (float(p[0, 0]), float(p[1, 0]))
        return self.pred

    def update(self, det):
        x1, y1, x2, y2, cls = det
        z = np.array([[(x1 + x2) / 2.0], [(y1 + y2) / 2.0]], np.float32)
        s = self.kf.correct(z)
        # smooth box size, keep latest class
        self.w = 0.7 * self.w + 0.3 * (x2 - x1)
        self.h = 0.7 * self.h + 0.3 * (y2 - y1)
        self.cls = cls
        self.pred = (float(s[0, 0]), float(s[1, 0]))
        self.hits += 1
        self.missed = 0
        self.detected_now = True
        self.trail.append((int(self.pred[0]), int(self.pred[1])))

    def coast(self):
        """No detection this frame: rely on Kalman prediction only."""
        # Keep the posterior = prior so the next predict() continues from it
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        self.missed += 1
        self.detected_now = False
        self.trail.append((int(self.pred[0]), int(self.pred[1])))

    def box(self):
        cx, cy = self.pred
        return (int(cx - self.w / 2), int(cy - self.h / 2),
                int(cx + self.w / 2), int(cy + self.h / 2))

# Tracker (association + track management)
class MultiTracker:
    def __init__(self, max_missed=15, min_hits=3, max_dist=80):
        self.tracks = []
        self.max_missed = max_missed   # delete after this many missed frames
        self.min_hits = min_hits       # confirm track after this many hits
        self.max_dist = max_dist       # gating distance in pixels

    def step(self, detections):
        # 2. predict all tracks
        for t in self.tracks:
            t.predict()

        # 3. association: cost = Euclidean distance(predicted, detection)
        matches, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), \
            list(range(len(detections)))
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    dc = ((d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0)
                    cost[i, j] = np.hypot(t.pred[0] - dc[0], t.pred[1] - dc[1])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_dist:      # gating
                    matches.append((r, c))
            unmatched_t = [i for i in unmatched_t if i not in [m[0] for m in matches]]
            unmatched_d = [j for j in unmatched_d if j not in [m[1] for m in matches]]

        # 4. update matched tracks
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        # 5. coast unmatched tracks (missing detection / occlusion)
        for ti in unmatched_t:
            self.tracks[ti].coast()
        # 6. new tracks
        for di in unmatched_d:
            self.tracks.append(Track(detections[di]))
        # 7. delete stale tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def confirmed(self):
        return [t for t in self.tracks if t.hits >= self.min_hits]

# Visualization
def color_for(track_id):
    rng = np.random.RandomState(track_id * 7919)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def draw(frame, tracks, frame_no):
    for t in tracks:
        col = color_for(t.id)
        x1, y1, x2, y2 = t.box()
        style = 2 if t.detected_now else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, style)
        status = "" if t.detected_now else " (pred)"
        label = f"ID {t.id} {t.cls}{status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # predicted/estimated position + trail
        cv2.circle(frame, (int(t.pred[0]), int(t.pred[1])), 4, (0, 0, 255), -1)
        pts = list(t.trail)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(frame, a, b, col, 2)
    cv2.putText(frame, f"Frame {frame_no}  Tracks: {len(tracks)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def parse_args():
    p = argparse.ArgumentParser(description="Multi-vehicle Kalman tracker")
    p.add_argument("--input", required=True, help="path to input video")
    p.add_argument("--output", default="output_tracked.mp4")
    p.add_argument("--detector", choices=["yolo", "motion"], default="yolo")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--max-missed", type=int, default=15,
                   help="frames a track survives without detection")
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-dist", type=float, default=80,
                   help="max pixel distance for association")
    p.add_argument("--min-area", type=int, default=800,
                   help="min blob area (motion detector)")
    p.add_argument("--show", action="store_true", help="show live window")
    return p.parse_args()


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    if a.detector == "yolo":
        try:
            detector = YoloDetector(a.weights, a.conf)
        except Exception as e:
            print(f"[warn] YOLO unavailable ({e}); using motion detector.")
            detector = MotionDetector(a.min_area)
    else:
        detector = MotionDetector(a.min_area)

    tracker = MultiTracker(a.max_missed, a.min_hits, a.max_dist)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracker.step(detector.detect(frame))
        frame = draw(frame, tracker.confirmed(), n)
        out.write(frame)
        if a.show:
            cv2.imshow("Kalman Vehicle Tracking (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. {n} frames processed. Saved to {a.output}")


if __name__ == "__main__":
    main()
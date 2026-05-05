"""
YOLO + BoT-SORT Person Tracking System

This program uses a YOLO object detection model combined with a BoT-SORT tracker
to detect and track a single person from a live webcam feed.

Main behavior:
- Captures frames from the laptop camera in real time.
- Detects people using YOLO (class 0 = person).
- Tracks detected people across frames using BoT-SORT tracking.
- Assigns a persistent ID to each detected person while they remain in view.
- Selects one "target person" (first detected person by default).
- Continuously follows and outputs the position (x-center) of the target person.
- Resets the target if the person is lost for more than 30 frames.

Dependencies:
- ultralytics (YOLO + tracking)
- OpenCV (video capture and display)
"""
class Tracker:
    def __init__(self, max_frames=30):
        self.target_id = -1
        self.frame_count = 0
        self.max_frames = max_frames

    def update(self, results, max_frames=30):
        x = 200.0  # default x-coordinate if no target is found
        found = False

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.tolist()
            boxes = results[0].boxes.xyxy.tolist()

            if self.target_id == -1 and len(ids) > 0:
                self.target_id = ids[0]

            for box, track_id in zip(boxes, ids):
                if track_id == self.target_id:
                    found = True
                    self.frame_count = 0

                    x = (box[0] + box[2]) / 2
                    print(f"ID: {track_id}, x: {x}")
                    break
        if not found:
          self.frame_count += 1
          print(f"Frame count since last seen: {self.frame_count}")
          if self.frame_count > self.max_frames:
              self.target_id = -1
              print("No objects detected for a while, resetting target ID")

        return x, self.target_id
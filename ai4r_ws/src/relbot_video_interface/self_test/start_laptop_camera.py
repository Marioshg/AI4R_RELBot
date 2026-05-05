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

import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
target_id = -1
MAX_FRAMES = 30 # Number of frames to wait before resetting target ID after losing track
frame_count = 0
found = False
while True:
    # Convert raw buffer to numpy array [height, width, channels]
    #frame = np.frombuffer(mapinfo.data, np.uint8).reshape(height, width, 3)
    
    ret, frame = cap.read()
    if not ret:
        break


    # Display the raw input frame for debugging
    cv2.imshow('Input Stream', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    #result = model(frame, conf=0.5, classes=[0])
    result = model.track(frame, persist=True,tracker="botsort.yaml")

    annotated = result[0].plot()
    x = 200.0  # object center x-coordinate
    found = False
    if result[0].boxes.id is not None:
        ids = result[0].boxes.id.tolist()
        boxes = result[0].boxes.xyxy.tolist()
        if(target_id == -1 and len(ids) > 0):
            target_id = ids[0]
        for box, track_id in zip(boxes, ids):
            if track_id == target_id:
                found = True
                frame_count = 0
                x = (box[0] + box[2]) / 2
                print(f"ID: {track_id}, x: {x}")
                break
    if not found:
        frame_count += 1
        print("Target left the Frame")
        print(f"Frame count since last seen: {frame_count}")
        if frame_count > MAX_FRAMES and target_id != -1:
            target_id = -1
            print("No objects detected for a while, resetting target ID")
        

    cv2.imshow('YOLO', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))




    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
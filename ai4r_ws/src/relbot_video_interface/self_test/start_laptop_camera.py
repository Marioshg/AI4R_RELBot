import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    # Perform object detection on an image
    results = model(
        frame,  # input image
        conf=0.5,  # confidence threshold (0-1)
         # NMS IoU threshold (0-1) for filtering overlapping boxes
    )

    # Visualize the results
    #for result in results:
    #    result.show()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
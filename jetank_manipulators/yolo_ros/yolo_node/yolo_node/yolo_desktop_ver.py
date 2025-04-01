from ultralytics import YOLO
import cv2

#Load model
model = YOLO("best.pt")

#Load classes
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Example to test
cap = cv2.VideoCapture("Jetracer pro AI Jetson Nano.road_following.ipynb.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Objectdetection
    results = model(frame)

    #Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
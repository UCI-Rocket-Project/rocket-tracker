from ultralytics import YOLO
import cv2 as cv

video = cv.VideoCapture('cycling.webm')

model = YOLO("yolov8n.pt")

tracked_id = None
while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model.track(frame, verbose=False, persist=True)[0]

    for conf, cls, xyxy, id in zip(results.boxes.conf, results.boxes.cls, results.boxes.xyxy, results.boxes.id):
        if id is None:
            continue
        x1, y1, x2, y2 = xyxy.int().tolist()
        cx = 602
        cy = 359
        if tracked_id is None and x1<cx<x2 and y1<cy<y2:
            tracked_id = id
            cv.imwrite("frame1.jpg", frame)
            
        if id == tracked_id:
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"ID: {id}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break
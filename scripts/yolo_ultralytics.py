import numpy as np
import cv2 as cv
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

if not torch.cuda.is_available():
    print("cuda unvailable")

cap = cv.VideoCapture(0)
model = YOLO("./models/yolo26n.pt")
model.to("cuda")
print(model.device)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results=model(frame, verbose=False)

    for r in results:
        boxes = r.boxes
        names = r.names  # class id → name mapping
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            if label == "cup":
                print("Cup detected!")
    result_frame=results[0].plot()
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', result_frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
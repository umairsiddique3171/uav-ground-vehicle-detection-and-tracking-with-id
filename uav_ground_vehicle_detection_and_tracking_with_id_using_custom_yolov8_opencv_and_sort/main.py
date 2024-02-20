# import libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from ultralytics import YOLO
import cv2
import json
import numpy as np
from sort.sort import *

mot_tracker = Sort()

# load model
model = YOLO("last.pt")

# load frames
cap = cv2.VideoCapture('sample.mp4')

# read frames
ret = True
frame_number = 0
while ret:
    ret, frame = cap.read()
    frame_number += 1
    if cv2.waitKey(25) & 0xFF == ord('n'):
        break

    # detections
    detections  = model(frame)   
    detections_ = []
    for detection in detections[0].boxes.data.tolist():
        if detection is not None: 
            x1, y1, x2, y2, score, class_id = detection
            if score > 0.65:
                detections_.append([x1, y1, x2, y2, score])


    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))
    print("---------------------------------------------------------------------------------")
    print("Frame:", frame_number)
    for track in track_ids:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, str(track_id), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print("Track ID:", track_id)
    print("---------------------------------------------------------------------------------")


    # show results
    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
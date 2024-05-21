"""
This script was used to collect data through a Raspberry Pi camera on a fixed-wing UAV during flight. Although model performance was trained beforehand on data grabbed from kaggle and other resources. This additional data collection was intended to enhance the model's performance by capturing aerial imagery that reflects the data distribution the model would encounter. This process was essential for optimizing the model's accuracy and efficiency.
"""

import cv2
import os
import time

data_dir = os.path.join(os.getcwd(),'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("Waiting for 1 minute before starting the capture...")
time.sleep(60)
print("Starting capture...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
saved_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame_count += 1
        
        if frame_count % 25 == 0:
            img_name = os.path.join(data_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(img_name, frame)
            print(f'Saved {img_name}')
            saved_count += 1
        
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

cap.release()
cv2.destroyAllWindows()

import cv2
import os
import time

data_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("Waiting for 1 minute before starting the capture...")
time.sleep(60)
print("Starting capture...")

frame_size = 640
capture_duration = 5 * 60 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

output_video = cv2.VideoWriter(os.path.join(data_dir, 'results.mp4'),
                               cv2.VideoWriter_fourcc(*'MP4V'),
                               20,
                               (frame_size, frame_size))

start_time = time.time()

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        resized_frame = cv2.resize(frame, (frame_size, frame_size))
        output_video.write(resized_frame)

        cv2.imshow('Frame', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('n'):
            print("Capture interrupted by user.")
            break

        if time.time() - start_time > capture_duration:
            print("Capture duration reached 5 minutes.")
            break

except KeyboardInterrupt:
    print("\nExiting due to keyboard interrupt...")

cap.release()
output_video.release()
cv2.destroyAllWindows()

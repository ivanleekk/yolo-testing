from ultralytics import YOLO

import cv2
cam = cv2.VideoCapture(1)
print(cam)

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    results = model(frame, device='mps')
    res_plotted = results[0].plot()
    cv2.imshow("result", res_plotted)
    
    if cv2.waitKey(1) == ord('q'):
        print("q pressed")
        break

cam.release()
cv2.destroyAllWindows()


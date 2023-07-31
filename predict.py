from ultralytics import YOLO

import cv2

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model
img = cv2.imread("bus.jpg")
results = model.predict(img)


# res = model(img)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey()


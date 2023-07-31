from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
if __name__ == '__main__':   
    model.train(data='coco128.yaml', epochs=3, imgsz=640, device='mps')

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/root/autodl-tmp/STN-YOLO/antenna/fine_tuning/antenna/YOLOV8N/Run_14/weights/best.pt")

# Define path to the image file
source = "/root/autodl-tmp/STN-YOLO/an (6).jpg"

# Run inference on the source
results = model(source,save=True)  # list of Results objects
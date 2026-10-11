from ultralytics import YOLO

# Train the model
# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# results = model.train(data="/root/autodl-tmp/STN-YOLO/ultralytics/cfg/datasets/coco128.yaml", epochs=100, imgsz=640)

# model = YOLO("/root/autodl-tmp/STN-YOLO/Saved_Models/feature_extract/plantdoc/YOLOV8L/Run_1/weights/best.pt")  # load a pretrained model (recommended for training)
model = YOLO("/root/autodl-tmp/STN-YOLO/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
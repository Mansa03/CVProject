from ultralytics import YOLO

#model
model = YOLO("YOLOv8n.yaml")

# Use the model
results = model.train(data="config.yaml", epochs=10)  # train the model
from ultralytics import YOLO

configpath = "C:/Users/razva/OneDrive - Viken fylkeskommune/Backup/Dokumenter/GitHub/YoloV8/Facial Recognition/config.yaml"

# Load a model
model = YOLO("yolov8n.yaml") # Build a new model from scratch

# Use the model
results = model.train(data=configpath, epochs=1) # train the model
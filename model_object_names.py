from ultralytics import YOLO

model = YOLO("./Model/visdrone-yolo11n.pt")
print(model.names)
from ultralytics import YOLOv10

model = YOLOv10("ultralytics/cfg/models/v10/yolov10s.yaml")
model.train(
    data="/home/la/CODE/代码/Orange_Pi/YOLOV10/yolov10-main/final/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    workers=4,
    device=0,
    amp=False,
    
)
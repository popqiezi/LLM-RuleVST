import torch
from ultralytics.nn.tasks import DetectionModel

model = DetectionModel("ultralytics/cfg/models/v10/yolov10s.yaml", nc=1, verbose=False)
model.train()

x = torch.randn(2, 3, 640, 640)
y = model(x)

print(type(y))
print(y.keys())
print(y["one2many"].keys())
print(len(y["one2many"]["feats"]), len(y["one2many"]["offsets"]))

for i, feat in enumerate(y["one2many"]["feats"]):
    print(f"feat[{i}] shape =", feat.shape)

for i, off in enumerate(y["one2many"]["offsets"]):
    print(f"offset[{i}] shape =", off.shape)
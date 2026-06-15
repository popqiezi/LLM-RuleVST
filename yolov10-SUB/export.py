from ultralytics import YOLOv10
from pathlib import Path

# =========================
# 1. 路径配置
# =========================
PT_MODEL_PATH = "/home/la/runs/detect/train2/weights/best.pt"
ONNX_OUT_PATH = "/home/la/CODE/代码/Orange_Pi/cfg/models/best2.onnx"

# =========================
# 2. 导出参数
# =========================
IMGSZ = 640
DEVICE = 0          # GPU导出可用0；CPU导出可改成 "cpu"
SIMPLIFY = False    # 如已安装 onnxsim 可改 True
DYNAMIC = False     # 固定尺寸导出更稳
OPSET = 12


def main():
    pt_path = Path(PT_MODEL_PATH)
    if not pt_path.exists():
        raise FileNotFoundError(f"模型不存在: {pt_path}")

    model = YOLOv10(str(pt_path))

    exported_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        device=DEVICE,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        opset=OPSET,
    )

    exported_path = Path(exported_path)

    # 如果你想固定输出文件名
    target_path = Path(ONNX_OUT_PATH)
    if exported_path.resolve() != target_path.resolve():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(target_path)
        print(f"导出成功: {target_path}")
    else:
        print(f"导出成功: {exported_path}")


if __name__ == "__main__":
    main()
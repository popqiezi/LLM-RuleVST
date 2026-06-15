import os
from pathlib import Path
import cv2
from ultralytics import YOLOv10

# =========================
# 1. 路径配置
# =========================
IMAGE_DIR = "/home/la/CODE/Experiment/效果图"
MODEL_PATH = "/home/la/runs/detect/train2/weights/best.pt"
OUTPUT_DIR = "/home/la/CODE/Experiment/效果图/YOLO"

# =========================
# 2. 推理参数
# =========================
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = 0  # GPU:0；如果用CPU可改成 "cpu"
SAVE_LABEL = True

BOX_COLOR = (255, 0, 0)      # 框颜色 BGR
TEXT_BG_COLOR = (255, 0, 0)  # 文字底色
TEXT_COLOR = (0, 0, 0)       # 文字颜色
LINE_WIDTH = 8              # 框粗细
FONT_SCALE = 2             # 字体大小
FONT_THICKNESS = 3           # 字体粗细

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def draw_results(image, result, line_width=2):
    names = result.names
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return image

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy
        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, line_width)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = y1
        tx1 = x1
        tx2 = x1 + tw + 6

        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), TEXT_BG_COLOR, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return image


def main():
    image_dir = Path(IMAGE_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_dir}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    model = YOLOv10(MODEL_PATH)

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    )

    if len(image_files) == 0:
        print("未找到图片。")
        return

    print(f"共找到 {len(image_files)} 张图片，开始推理...")

    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[跳过] 读取失败: {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        vis = image.copy()
        vis = draw_results(vis, result, line_width=LINE_WIDTH)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), vis)

        if SAVE_LABEL:
            txt_path = output_dir / f"{img_path.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        f.write(
                            f"{cls_id} {xyxy[0]:.2f} {xyxy[1]:.2f} {xyxy[2]:.2f} {xyxy[3]:.2f} {conf:.4f}\n"
                        )

        print(f"[{idx}/{len(image_files)}] 已保存: {out_path}")

    print("全部完成。")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""用保存的关键帧测 YOLOE 检测能力"""
import cv2
import numpy as np
from ultralytics import YOLOE

FRAME = "/home/tony/wey/rebot_developer/wrc/runs/20260920_230757/keyframes/0000_get_observation_after.jpg"
MODEL = "/home/tony/wey/rebot_developer/wrc/models/yoloe-26s-seg.pt"
CLASSES = ["red cube", "cube", "box", "banana", "bottle", "cup", "toy", "pen", "ball"]

img = cv2.imread(FRAME)
if img is None:
    raise SystemExit(f"cannot read {FRAME}")
print(f"image: {img.shape}  mean={img.mean():.1f}")

model = YOLOE(MODEL)
model.set_classes(CLASSES)

# 用低阈值看有没有候选
for conf in [0.05, 0.10, 0.20]:
    results = model.predict(img, conf=conf, verbose=False)
    r = results[0]
    n = len(r.boxes) if r.boxes is not None else 0
    print(f"\n--- conf={conf}: {n} detection(s) ---")
    if n > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            name = CLASSES[cls_id] if cls_id < len(CLASSES) else "?"
            score = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()
            print(f"  {name:12s} conf={score:.2f} bbox=({xyxy[0]:.0f},{xyxy[1]:.0f})-({xyxy[2]:.0f},{xyxy[3]:.0f})")
        # 保存可视化结果
        out = f"/tmp/yoloe_result_conf{int(conf*100)}.jpg"
        cv2.imwrite(out, r.plot())
        print(f"  saved -> {out}")
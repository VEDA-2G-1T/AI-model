from ultralytics import YOLO

# ❶ 파인튜닝된 모델 로드
model = YOLO('./model/model.pt')

# ❷ FULL INT8 TFLite export
model.export(
    format='tflite',
    int8=True,        # int8 quantization ON
    dynamic=False,    # Dynamic range(False) → Static (full‑integer)
    imgsz=192,         # 입력 해상도,
    data='fall.yaml'
)

print("Full‑Integer INT8 TFLite 모델 변환 완료!")

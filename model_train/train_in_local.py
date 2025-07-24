# train_sh17.py

import os
# ─── OpenMP 충돌 방지 ─────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]    = "1"

import torch
from ultralytics import YOLO

def update_paths(file_path: str, root: str, out_dir: str) -> str:
    """
    train_files.txt, val_files.txt 의 상대경로를 절대경로로 바꿔
    updated_<원본파일명>.txt 로 저장하고, 새 경로를 반환.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    updated = [os.path.normpath(os.path.join(root, rel)) + "\n" for rel in lines]
    new_name = f"updated_{os.path.basename(file_path)}"
    new_path = os.path.join(out_dir, new_name)
    with open(new_path, 'w', encoding='utf-8') as f:
        f.writelines(updated)
    print(f"[OK] {new_path} ({len(updated)} entries)")
    return new_path

def main():
    # 1) 경로 설정
    LOCAL_ROOT      = r'E:/sh17'
    IMAGE_ROOT      = os.path.join(LOCAL_ROOT, 'images')
    TRAIN_TXT       = os.path.join(LOCAL_ROOT, 'train_files.txt')
    VAL_TXT         = os.path.join(LOCAL_ROOT, 'val_files.txt')
    WORKING_DIR     = os.path.join(LOCAL_ROOT, 'working')
    YAML_LOCAL_PATH = os.path.join(LOCAL_ROOT, 'sh17.yaml')
    PROJECT_DIR     = os.path.join(LOCAL_ROOT, 'yolo_runs')

    # 2) train/val 목록 절대경로로 변환
    update_paths(TRAIN_TXT, IMAGE_ROOT, WORKING_DIR)
    update_paths(VAL_TXT,   IMAGE_ROOT, WORKING_DIR)

    # 3) CUDA 사용 가능 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 찾을 수 없습니다. 드라이버/라이브러리 설치를 확인하세요.")
    print("CUDA available:", torch.cuda.is_available(),
          "| Device:", torch.cuda.get_device_name(0))

    # 4) 모델 로드 및 GPU 이동
    model = YOLO('yolo11n.pt').to('cuda:0')

    # 5) 학습 실행
    results = model.train(
        data      = YAML_LOCAL_PATH,
        epochs    = 100,
        batch     = 4,              # VRAM 상황에 따라 1–2로 조정
        imgsz     = 640,            # 
        device    = 'cuda:0',       # 명시적으로 CUDA
        project   = PROJECT_DIR,
        name      = 'v11n_exp1',
        exist_ok  = True,
        plots     = False,          # labels.jpg 생성 스킵
        workers   = 0,              # DataLoader 멀티프로세싱 비활성화
        cache     = False,          # cache 비활성화
        mosaic    = False,          # augmentation 비활성화
        mixup     = False
    )
pt -> fp32 -> int8
    # 6) 검증
    metrics = model.val(
        data   = YAML_LOCAL_PATH,
        imgsz  = 640,
        batch  = 4,
        device = 'cuda:0',
        cache  = False
    )
    print("Validation metrics:", metrics)

    # 7) ONNX 변환
    onnx_path = model.export(
        format   = 'onnx',
        imgsz    = 640,
        simplify = True
    )
    print("ONNX saved at:", onnx_path)

if __name__ == '__main__':
    main()


'''

100 epochs completed in 60.765 hours.
Optimizer stripped from E:\sh17\yolo_runs\v8n_exp1\weights\last.pt, 6.2MB
Optimizer stripped from E:\sh17\yolo_runs\v8n_exp1\weights\best.pt, 6.2MB

Validating E:\sh17\yolo_runs\v8n_exp1\weights\best.pt...
Ultralytics 8.3.53 🚀 Python-3.9.19 torch-2.5.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4090, 24564MiB)
Model summary (fused): 168 layers, 3,008,963 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 203/203 [06:
                   all       1620      15358        0.6      0.486      0.514      0.324
                person       1515       2734      0.827      0.854      0.879      0.694
                   ear        987       1612      0.853      0.679      0.725      0.434
              ear-mufs         38         49      0.428      0.168       0.19      0.109
                  face       1155       1855      0.919      0.848      0.892      0.657
            face-guard         23         24      0.328      0.292      0.322      0.202
             face-mask         75        151       0.76      0.603      0.635      0.357
                  foot         64        149       0.25     0.0359     0.0757     0.0351
                  tool        455        923      0.378       0.25      0.221      0.107
               glasses        323        398      0.637      0.553      0.555      0.269
                gloves        254        529       0.57      0.395      0.426      0.253
                helmet         93        154      0.663      0.519      0.573      0.381
                 hands       1284       3212      0.806      0.768      0.811      0.532
                  head       1314       2427      0.911      0.847      0.887      0.677
          medical-suit         30         43      0.447      0.442      0.441      0.205
                 shoes        320        956      0.633      0.446      0.498      0.259
           safety-suit         28         45      0.328      0.156      0.216      0.133
           safety-vest         45         97      0.457      0.402      0.385      0.205
Speed: 0.5ms preprocess, 1.7ms inference, 0.0ms loss, 1.3ms postprocess per image
Ultralytics 8.3.53 🚀 Python-3.9.19 torch-2.5.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4090, 24564MiB)
Model summary (fused): 168 layers, 3,008,963 parameters, 0 gradients, 8.1 GFLOPs
val: Scanning E:\sh17\labels.cache... 1620 images, 0 backgrounds, 0 corrupt: 100%|██████████| 1620/1620 [00:00<?, ?it/s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 405/405 [07:
                   all       1620      15358      0.606      0.484      0.515      0.324
                person       1515       2734      0.829      0.853       0.88      0.694
                   ear        987       1612      0.854      0.678      0.723      0.436
              ear-mufs         38         49      0.502      0.185      0.211      0.113
                  face       1155       1855       0.92      0.846      0.892      0.657
            face-guard         23         24      0.347      0.292      0.333       0.21
             face-mask         75        151      0.763      0.603      0.634      0.358
                  foot         64        149      0.244     0.0348     0.0756     0.0352
                  tool        455        923      0.369      0.246      0.219      0.106
               glasses        323        398      0.636      0.548      0.553      0.269
                gloves        254        529      0.581      0.397      0.427      0.254
                helmet         93        154      0.677      0.519      0.574      0.382
                 hands       1284       3212      0.809      0.767      0.812      0.533
                  head       1314       2427      0.913      0.846      0.888      0.678
          medical-suit         30         43      0.453      0.442      0.439      0.198
                 shoes        320        956      0.641      0.446      0.501      0.259
           safety-suit         28         45      0.288      0.133      0.205       0.13
           safety-vest         45         97       0.47      0.392      0.387      0.204
Speed: 0.7ms preprocess, 2.9ms inference, 0.0ms loss, 1.4ms postprocess per image
Validation metrics: ultralytics.utils.metrics.DetMetrics object with attributes:
'''
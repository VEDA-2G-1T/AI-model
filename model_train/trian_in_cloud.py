# 만약 ultralytics나 torch가 없다면—한 번만 실행
!pip install --upgrade ultralytics torch torchvision
!pip install ultralytics
# 1) 기존 충돌 가능성 있는 OpenCV 제거
!pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# 2) Headless 버전 설치
!pip install opencv-python-headless

import os
import torch
from ultralytics import YOLO

# ─── OpenMP 충돌 방지 ─────────────────────────────────────────
# 주피터 노트북 환경에서는 이 설정이 필수적이지 않을 수 있으나,
# 만약을 위해 유지하는 것이 좋습니다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]     = "1"


def update_paths(file_path: str, root: str, out_dir: str) -> str:
    """
    train_files.txt, val_files.txt 의 상대경로를 절대경로로 바꿔
    updated_<원본파일명>.txt 로 저장하고, 새 경로를 반환.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    updated_lines = [os.path.normpath(os.path.join(root, rel_path)) + "\n" for rel_path in lines]
    
    new_name = f"updated_{os.path.basename(file_path)}"
    new_path = os.path.join(out_dir, new_name)
    
    with open(new_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
        
    print(f"[OK] 새 경로 파일 생성: {new_path} ({len(updated_lines)}개 항목)")
    return new_path
    
    # ⭐️ 중요: 이 경로는 사용자의 실제 데이터셋 위치에 맞게 수정해주세요.
LOCAL_ROOT      = '/home/elicer/SH17' 

# 아래 경로는 LOCAL_ROOT를 기준으로 자동 설정됩니다.
IMAGE_ROOT      = os.path.join(LOCAL_ROOT, 'images')
TRAIN_TXT       = os.path.join(LOCAL_ROOT, 'train_files.txt')
VAL_TXT         = os.path.join(LOCAL_ROOT, 'val_files.txt')
WORKING_DIR     = os.path.join(LOCAL_ROOT, 'working')
YAML_LOCAL_PATH = os.path.join(LOCAL_ROOT, 'sh17.yaml')
PROJECT_DIR     = os.path.join(LOCAL_ROOT, 'yolo_runs')

print(f"프로젝트 기본 경로: {LOCAL_ROOT}")
print(f"결과 저장 경로: {PROJECT_DIR}")
# train/val 목록의 상대경로를 절대경로로 변환하여 새 파일을 생성합니다.
# 이 파일들은 sh17.yaml 파일 내에서 참조되어야 합니다.
# (참고: Ultralytics v8+ 에서는 yaml 파일 내에서 자동으로 경로를 처리해주기도 하지만,
# 명시적으로 절대경로를 만들어주면 경로 관련 오류를 방지할 수 있습니다.)
update_paths(TRAIN_TXT, IMAGE_ROOT, WORKING_DIR)
update_paths(VAL_TXT,   IMAGE_ROOT, WORKING_DIR)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA를 찾을 수 없습니다. 드라이버/라이브러리 설치를 확인하세요.")

print("CUDA 사용 가능:", torch.cuda.is_available())
print("사용 중인 GPU:", torch.cuda.get_device_name(0))
# 'yolo11n.pt' 모델을 로드하고 CUDA 장치로 보냅니다.
model = YOLO('yolo11n.pt').to('cuda:0')
print("모델 로드 완료: yolo11n.pt")

# 모델 학습을 실행합니다.
results = model.train(
    data       = YAML_LOCAL_PATH,
    epochs     = 100,
    batch      = 4,        # VRAM(GPU 메모리) 부족 시 1 또는 2로 줄여야 합니다.
    imgsz      = 320,
    device     = 'cuda:0', # 사용할 GPU 장치를 명시적으로 지정
    project    = PROJECT_DIR,
    name       = 'v11n_exp1',
    exist_ok   = True,     # 동일한 이름의 실험 폴더가 있어도 덮어쓰기 허용
    plots      = True,     # 학습 과정 시각화 파일(confusion_matrix.png, results.png 등) 생성
    workers    = 0,        # DataLoader 멀티프로세싱 비활성화 (Windows + 주피터 환경의 안정성)
    cache      = False,    # 이미지 캐싱 비활성화
    mosaic     = False,    # Mosaic augmentation 비활성화
    mixup      = False     # Mixup augmentation 비활성화
)

print("학습이 완료되었습니다.")


# 학습된 최종 모델로 검증을 실행합니다.
metrics = model.val(
    data   = YAML_LOCAL_PATH,
    imgsz  = 320,
    batch  = 4,
    device = 'cuda:0',
    cache  = False
)

# 검증 결과 출력
# print(metrics) # 전체 메트릭 객체 출력
print("\n--- 검증 결과 ---")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   mAP75: {metrics.box.map75:.4f}")

# 모델을 ONNX 형식으로 변환합니다.
onnx_path = model.export(
    format   = 'onnx',
    imgsz    = 320,
    simplify = True  # ONNX 모델 구조를 단순화하여 추론 속도 향상
)

print(f"모델이 ONNX 형식으로 변환되어 아래 경로에 저장되었습니다:\n{onnx_path}")
from ultralytics import YOLO

# 1) .pt 파일 로드
model = YOLO('best11.pt')  

# 2) 검증 실행 (data는 학습에 사용한 YAML 파일)
#    imgsz: 입력 이미지 크기, batch: GPU 메모리에 맞춰 조정
results = model.val(
    data   = '"C:/Users/wjdgu/Downloads/vest-no-vest.v1i.yolov11/data.yaml"',
    imgsz  = 640,
    batch  = 4,
    device = 'cuda:0'
)

# 3) 주요 지표 출력
print(f"Precision: {results.metrics[0]:.3f}")
print(f"Recall:    {results.metrics[1]:.3f}")
print(f"mAP@0.5:   {results.metrics[2]:.3f}")
print(f"mAP@0.5-0.95: {results.metrics[3]:.3f}")
# 혼동행렬 생성 및 시각화
cm = results.plot_confusion_matrix(normalize=True)  
# Jupyter에선 바로 출력됩니다.
# PR curve (Precision-Recall) 그리기
pr = results.plot_pr_curve()  
import pandas as pd

# 클래스별 AP 정보(예: yolov8에서 results.box.map50.tolist() 가능)
classes    = model.names.values()        # 클래스 이름 리스트
map50_list = results.box.map50.tolist()  # mAP@0.5 per class

df = pd.DataFrame({
    'class': classes,
    'mAP@0.5': map50_list
}).sort_values('mAP@0.5', ascending=False)

df.style.format({'mAP@0.5': '{:.3f}'})

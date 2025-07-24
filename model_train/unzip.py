import zipfile
from pathlib import Path
from tqdm import tqdm  # tqdm 라이브러리를 임포트합니다.

# --------------------------------------------------------------------------
# 만약 tqdm이 설치되지 않았다면, 터미널이나 주피터 노트북 셀에서
# !pip install tqdm
# 명령어를 실행하여 설치해주세요.
# --------------------------------------------------------------------------

zip_path = Path('/home/elicer/SH17.zip')
dest_dir = Path('SH17')

# 대상 폴더가 없으면 생성
dest_dir.mkdir(parents=True, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # zf.infolist()로 전체 파일 목록을 가져옵니다.
        members = zf.infolist()
        
        # tqdm으로 파일 목록을 감싸주면, for문이 돌 때마다 진행률이 표시됩니다.
        for member in tqdm(members, desc=f"'{zip_path.name}' 압축 해제 중", unit="개"):
            zf.extract(member, dest_dir)

    print(f"\n압축 해제 완료: {dest_dir.resolve()}")

except FileNotFoundError:
    print(f"[오류] 파일을 찾을 수 없습니다: {zip_path}")
except zipfile.BadZipFile:
    print(f"[오류] '{zip_path.name}'은(는) 유효한 ZIP 파일이 아닙니다.")
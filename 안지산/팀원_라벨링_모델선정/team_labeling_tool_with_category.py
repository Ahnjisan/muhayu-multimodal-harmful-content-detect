# -*- coding: utf-8 -*-
"""
무하유 팀원용 데이터 라벨링 도구 (카테고리 분류)
===============================================================

이 스크립트는 팀원들이 수집한 이미지/비디오 데이터를 **완전 수동으로** 카테고리별로 라벨링하는 도구입니다.

주요 특징:
  - YOLO 자동 탐지 없음 (순수 수동 라벨링)
  - 단순 유해/안전 분류 → 10개 카테고리 세분화
  - 최종 독립 평가용 데이터 라벨링 (1200개 = 인당 400개 × 3명)
  - 각자 이름 폴더에 결과 저장

데이터 구성 (인당 400개):
  - 안전 이미지: 100개
  - 유해 이미지: 100개
  - 안전 비디오: 100개
  - 유해 비디오: 100개

사용 방법:
1. 팀원 이름 수정
2. python team_labeling_tool_with_category.py 실행
3. 화면에 나타나는 파일을 보고 0-9 키로 카테고리 선택

폴더 구조:
  무하유_유해콘텐츠_데이터_모델선정/
  ├── 2_실제_수집_데이터/
  │   ├── 안지산/
  │   │   ├── 이미지/              # 유해 이미지
  │   │   ├── 안전_이미지/         # 안전 이미지
  │   │   ├── 비디오/              # 유해 비디오
  │   │   └── 안전_비디오/         # 안전 비디오
  │   ├── 박상원/ (동일)
  │   └── 임영재/ (동일)
  └── 3_라벨링_파일/
      ├── 안지산/
      │   └── 안지산_labels_categorized.json
      ├── 박상원/
      │   └── 박상원_labels_categorized.json
      └── 임영재/
          └── 임영재_labels_categorized.json

출력 파일:
  - 3_라벨링_파일/{이름}/{이름}_labels_categorized.json

카테고리 목록 (10개):
  1. weapons     - 무기 (knife, gun, sword 등)
  2. violence    - 폭력 (fighting, assault 등)
  3. alcohol     - 음주 (drinking, drunk 등)
  4. smoking     - 흡연 (cigarette, tobacco 등)
  5. drugs       - 약물 (drug use, syringe 등)
  6. blood       - 혈액/상처 (injury, wound 등)
  7. threat      - 위협 (threatening, intimidation 등)
  8. sexual      - 성적 콘텐츠 (sexual violence 등)
  9. dangerous   - 위험행동 (self harm, reckless 등)
  0. safe        - 안전 (유해하지 않음)
  S. skip        - 건너뛰기 (판단 불가, 평가 제외)

⚠️ 중요: 폴더 이름 ≠ 최종 라벨
  - 폴더 이름: 수집 시 예상 분류
  - 수동 라벨: 실제로 본 정확한 판단 ✅
  - "비디오" 폴더 파일도 0(안전) 선택 가능
  - "안전_비디오" 폴더 파일도 1-9(유해) 선택 가능
  - 최종 평가는 수동 선택한 카테고리 기준!

주의사항:
  - opencv-python 설치 필요: pip install opencv-python
  - numpy 설치 필요: pip install numpy

"""

# ============================================================
# 필수 라이브러리 Import
# ============================================================
import os
import sys
import json
import cv2
import numpy as np
import hashlib
from pathlib import Path

# ============================================================
# 설정 부분 - 팀원이 수정해야 하는 부분
# ============================================================
# 팀원 이름 (출력 파일명에 사용)
TEAM_MEMBER_NAME = "안지산"  # ← 여기만 수정! 예: "박상원", "임영재", "안지산"

# 데이터 폴더 경로 설정
BASE_PATH = './무하유_유해콘텐츠_데이터_모델선정/'

# 데이터 폴더 (4개 폴더의 모든 파일)
DATA_DIRS = {
    '이미지': BASE_PATH + f'2_실제_수집_데이터/{TEAM_MEMBER_NAME}/이미지/',
    '안전_이미지': BASE_PATH + f'2_실제_수집_데이터/{TEAM_MEMBER_NAME}/안전_이미지/',
    '비디오': BASE_PATH + f'2_실제_수집_데이터/{TEAM_MEMBER_NAME}/비디오/',
    '안전_비디오': BASE_PATH + f'2_실제_수집_데이터/{TEAM_MEMBER_NAME}/안전_비디오/'
}

# 출력 폴더 및 파일
OUTPUT_DIR = BASE_PATH + f'3_라벨링_파일/{TEAM_MEMBER_NAME}/'
OUTPUT_FILE = OUTPUT_DIR + f'{TEAM_MEMBER_NAME}_labels_categorized.json'

# ============================================================
# 카테고리 정의 (10개)
# ============================================================
CATEGORIES = {
    '1': {'name': 'weapons', 'label': '무기', 'harmful': True},
    '2': {'name': 'violence', 'label': '폭력', 'harmful': True},
    '3': {'name': 'alcohol', 'label': '음주', 'harmful': True},
    '4': {'name': 'smoking', 'label': '흡연', 'harmful': True},
    '5': {'name': 'drugs', 'label': '약물', 'harmful': True},
    '6': {'name': 'blood', 'label': '혈액/상처', 'harmful': True},
    '7': {'name': 'threat', 'label': '위협', 'harmful': True},
    '8': {'name': 'sexual', 'label': '성적콘텐츠', 'harmful': True},
    '9': {'name': 'dangerous', 'label': '위험행동', 'harmful': True},
    '0': {'name': 'safe', 'label': '안전', 'harmful': False},
}

# ============================================================
# 파일 해시 계산 (중복 검출용)
# ============================================================
def get_file_hash(filepath):
    """
    파일의 MD5 해시값 계산 (중복 파일 검출용)
    
    Args:
        filepath: 파일 경로
        
    Returns:
        str: MD5 해시값
    """
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


# ============================================================
# 폴더 확인
# ============================================================
def check_folders():
    """필요한 폴더 존재 확인"""
    missing_folders = []
    
    for folder_name, folder_path in DATA_DIRS.items():
        if not os.path.exists(folder_path):
            missing_folders.append(folder_path)
    
    if missing_folders:
        print(f"❌ 다음 데이터 폴더가 없습니다:")
        for folder in missing_folders:
            print(f"   {folder}")
        return False
    
    # 출력 폴더 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    return True


# ============================================================
# 파일 목록 가져오기
# ============================================================
def get_all_files():
    """
    4개 데이터 폴더에서 모든 이미지/비디오 파일 가져오기
    파일 해시값도 계산하여 중복 검출에 사용
    
    Returns:
        list: 파일 정보 리스트 (경로, 타입, 이름, 폴더, 해시)
    """
    supported_extensions = {
        'image': ['.jpg', '.jpeg', '.png', '.bmp'],
        'video': ['.mp4', '.avi', '.mov', '.mkv']
    }
    
    all_files = []
    
    print("파일 목록 로딩 중 (중복 검출을 위해 해시값 계산)...")
    
    for folder_name, folder_path in DATA_DIRS.items():
        if not os.path.exists(folder_path):
            continue
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if not os.path.isfile(filepath):
                continue
            
            ext = os.path.splitext(filename)[1].lower()
            
            # 이미지 파일
            if ext in supported_extensions['image']:
                file_hash = get_file_hash(filepath)
                all_files.append({
                    'path': filepath,
                    'type': 'image',
                    'name': filename,
                    'folder': folder_name,  # 어느 폴더에서 왔는지
                    'hash': file_hash  # 중복 검출용 해시
                })
            # 비디오 파일
            elif ext in supported_extensions['video']:
                file_hash = get_file_hash(filepath)
                all_files.append({
                    'path': filepath,
                    'type': 'video',
                    'name': filename,
                    'folder': folder_name,
                    'hash': file_hash  # 중복 검출용 해시
                })
    
    return all_files


# ============================================================
# 기존 라벨 로드
# ============================================================
def load_existing_labels():
    """
    기존에 라벨링한 결과가 있으면 로드
    
    Returns:
        dict: 기존 라벨 딕셔너리
    """
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"✓ 기존 라벨 로드: {len(labels)}개")
            return labels
        except:
            pass
    
    return {}


# ============================================================
# 이미지 라벨링
# ============================================================
def label_image(filepath, filename, folder_name, file_hash, idx, total):
    """
    이미지 파일 라벨링
    
    Args:
        filepath: 이미지 파일 경로
        filename: 파일명
        folder_name: 폴더명 (이미지/안전_이미지)
        file_hash: 파일 해시값 (중복 검출용)
        idx: 현재 인덱스
        total: 전체 파일 개수
        
    Returns:
        dict: 라벨 정보 또는 None (건너뛰기/종료)
    """
    # 이미지 로드 (한글 경로 지원)
    img_array = cv2.imdecode(
        np.fromfile(filepath, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    
    if img_array is None:
        print(f"  ⚠️ 이미지 로드 실패: {filename}")
        return None
    
    # 이미지 복사
    display_img = img_array.copy()
    
    # 진행 상황 표시
    progress = f"[{idx+1}/{total}] {folder_name}/{filename}"
    cv2.putText(display_img, progress, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(display_img, "Press 0-9 for category, S:Skip, Q:Quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 이미지 크기 조정
    h, w = display_img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)))
    
    # 이미지 표시
    cv2.imshow('Manual Labeling - Press 0-9, S, Q', display_img)
    
    # 콘솔 출력
    print(f"\n[{idx+1}/{total}] {folder_name}/{filename}")
    
    # 사용자 입력 대기
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # Q: 종료
        if key in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            return 'QUIT'
        
        # S: 스킵
        elif key in [ord('s'), ord('S')]:
            print("  → 건너뛰기\n")
            cv2.destroyAllWindows()
            return None
        
        # 0-9: 카테고리 선택
        elif chr(key) in CATEGORIES.keys():
            category_key = chr(key)
            category_info = CATEGORIES[category_key]
            
            label_data = {
                'type': 'image',
                'source_folder': folder_name,
                'category': category_info['name'],
                'category_label': category_info['label'],
                'is_harmful': category_info['harmful'],
                'file_hash': file_hash  # 중복 검출용 해시
            }
            
            print(f"  ✓ 카테고리: {category_info['label']} ({category_info['name']})\n")
            cv2.destroyAllWindows()
            return label_data
        
        else:
            print("  ⚠️ 올바른 키를 입력하세요 (0-9, S, Q)")
    

# ============================================================
# 비디오 라벨링
# ============================================================
def label_video(filepath, filename, folder_name, file_hash, idx, total):
    """
    비디오 파일 라벨링
    
    Args:
        filepath: 비디오 파일 경로
        filename: 파일명
        folder_name: 폴더명 (비디오/안전_비디오)
        file_hash: 파일 해시값 (중복 검출용)
        idx: 현재 인덱스
        total: 전체 파일 개수
        
    Returns:
        dict: 라벨 정보 또는 None (건너뛰기/종료)
    """
    # 비디오 파일 열기
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"  ⚠️ 비디오 로드 실패: {filename}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # 중간 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2 if total_frames > 0 else 0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  ⚠️ 비디오 프레임 읽기 실패: {filename}")
        return None
    
    # 프레임에 정보 오버레이
    progress = f"[{idx+1}/{total}] {folder_name}/{filename}"
    cv2.putText(frame, progress, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, f"Duration: {duration:.1f}s | Press 0-9, S, Q", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 프레임 크기 조정
    h, w = frame.shape[:2]
    if w > 1000:
        frame = cv2.resize(frame, (1000, int(h * 1000 / w)))
    
    # 프레임 표시
    cv2.imshow('Manual Labeling - Press 0-9, S, Q', frame)
    
    # 콘솔 출력
    print(f"\n[{idx+1}/{total}] {folder_name}/{filename} (비디오: {duration:.1f}초)")
    
    # 사용자 입력 대기
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        # Q: 종료
        if key in [ord('q'), ord('Q')]:
            cv2.destroyAllWindows()
            return 'QUIT'
        
        # S: 스킵
        elif key in [ord('s'), ord('S')]:
            print("  → 건너뛰기\n")
            cv2.destroyAllWindows()
            return None
        
        # 0-9: 카테고리 선택
        elif chr(key) in CATEGORIES.keys():
            category_key = chr(key)
            category_info = CATEGORIES[category_key]
            
            label_data = {
                'type': 'video',
                'source_folder': folder_name,
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'category': category_info['name'],
                'category_label': category_info['label'],
                'is_harmful': category_info['harmful'],
                'file_hash': file_hash  # 중복 검출용 해시
            }
            
            print(f"  ✓ 카테고리: {category_info['label']} ({category_info['name']})\n")
            cv2.destroyAllWindows()
            return label_data
        
        else:
            print("  ⚠️ 올바른 키를 입력하세요 (0-9, S, Q)")


# ============================================================
# 메인 실행 함수
# ============================================================
def main():
    """
    전체 라벨링 프로세스 실행
    """
    print("\n" + "="*80)
    print(f"무하유 팀원용 데이터 라벨링 도구 (완전 수동) - {TEAM_MEMBER_NAME}")
    print("="*80)
    print("버전: 2.0 - 완전 수동 라벨링 (YOLO 없음)")
    print("="*80 + "\n")
    
    # 1. 폴더 확인
    if not check_folders():
        input("\nEnter를 눌러 종료...")
        return
    
    # 2. 파일 목록 가져오기
    print("파일 목록 로딩 중...")
    all_files = get_all_files()
    
    if not all_files:
        print(f"❌ 데이터 폴더에 파일이 없습니다.")
        input("\nEnter를 눌러 종료...")
        return
    
    # 폴더별 통계
    from collections import Counter
    folder_counts = Counter(f['folder'] for f in all_files)
    
    print(f"✓ 총 {len(all_files)}개 파일 발견")
    for folder_name, count in folder_counts.items():
        print(f"  - {folder_name}: {count}개")
    print()
    
    # 3. 기존 라벨 로드
    labels = load_existing_labels()
    
    # 기존 라벨링된 파일들의 해시값 세트 (중복 검출용)
    labeled_hashes = set()
    for label_data in labels.values():
        if 'file_hash' in label_data and label_data['file_hash']:
            labeled_hashes.add(label_data['file_hash'])
    
    print(f"✓ 기존 라벨링된 고유 파일: {len(labeled_hashes)}개 (중복 제외)\n")
    
    # 4. 카테고리 안내 출력
    print("="*80)
    print("카테고리 선택 가이드")
    print("="*80)
    for key, info in CATEGORIES.items():
        harmful_mark = "🔴" if info['harmful'] else "🟢"
        print(f"  {key}: {harmful_mark} {info['label']} ({info['name']})")
    print("\n  S: 건너뛰기 (이 파일 제외)")
    print("  Q: 검증 중단 및 종료")
    print("="*80 + "\n")
    
    input("준비되면 Enter를 눌러 시작...")
    
    # 5. 라벨링 시작
    print("\n라벨링 시작!\n")
    
    duplicate_count = 0
    
    for idx, file_info in enumerate(all_files):
        filename = file_info['name']
        filepath = file_info['path']
        file_type = file_info['type']
        folder_name = file_info['folder']
        file_hash = file_info['hash']
        
        # 이미 라벨링된 파일은 건너뛰기 (파일명 기준)
        if filename in labels:
            print(f"[{idx+1}/{len(all_files)}] {folder_name}/{filename} - ✓ 이미 라벨링됨 (건너뛰기)")
            continue
        
        # 중복 파일 검출 (해시 기준)
        if file_hash and file_hash in labeled_hashes:
            duplicate_count += 1
            print(f"[{idx+1}/{len(all_files)}] {folder_name}/{filename} - 🔁 중복 파일 (건너뛰기)")
            continue
        
        # 파일 타입에 따라 라벨링
        if file_type == 'image':
            result = label_image(filepath, filename, folder_name, file_hash, idx, len(all_files))
        else:  # video
            result = label_video(filepath, filename, folder_name, file_hash, idx, len(all_files))
        
        # 결과 처리
        if result == 'QUIT':
            print("\n라벨링 중단됨")
            break
        elif result is not None:
            labels[filename] = result
            # 해시값을 세트에 추가 (이후 중복 검출용)
            if file_hash:
                labeled_hashes.add(file_hash)
            
            # 중간 저장 (10개마다)
            if (idx + 1) % 10 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(labels, f, indent=2, ensure_ascii=False)
                print(f"  💾 중간 저장 완료 ({len(labels)}개)")
    
    cv2.destroyAllWindows()
    
    # 6. 최종 저장
    print("\n" + "="*80)
    print("최종 저장 중...")
    print("="*80 + "\n")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 라벨링 결과 저장: {OUTPUT_FILE}\n")
    
    # 7. 카테고리별 통계
    print("="*80)
    print("카테고리별 통계")
    print("="*80)
    
    from collections import Counter
    category_counts = Counter(item['category'] for item in labels.values())
    
    print(f"\n팀원: {TEAM_MEMBER_NAME}")
    print(f"전체: {len(labels)}개")
    print("-" * 80)
    
    for cat_key, cat_info in CATEGORIES.items():
        count = category_counts.get(cat_info['name'], 0)
        harmful_mark = "🔴" if cat_info['harmful'] else "🟢"
        print(f"  {harmful_mark} {cat_info['label']} ({cat_info['name']}): {count}개")
    
    harmful_total = sum(count for cat_name, count in category_counts.items() 
                       if cat_name != 'safe')
    
    print("-" * 80)
    print(f"유해: {harmful_total}개 | 안전: {category_counts.get('safe', 0)}개")
    if duplicate_count > 0:
        print(f"🔁 중복 파일 제외: {duplicate_count}개")
    print(f"\n✓ 출력 파일: {os.path.abspath(OUTPUT_FILE)}")
    print("="*80 + "\n")


# ============================================================
# 스크립트 실행
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        input("\nEnter를 눌러 종료...")

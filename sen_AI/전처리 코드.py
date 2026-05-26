import pandas as pd
import numpy as np
import json
import re

# 1. 파일 경로 (본인의 경로에 맞춰 수정)
meta_path = r"D:\SWPJ-4\전처리 완료(수정 완)\used_videos_final_CLEANED.csv"
classes_path = r"C:\Users\wolah\Desktop\전처리 완료 데이터(0521)\전처리 완료\processed\classes.npy"

# 2. 데이터 로드
meta_df = pd.read_csv(meta_path)
classes = np.load(classes_path, allow_pickle=True)

# 3. SEN ID 추출 정규식
def extract_sen(text):
    match = re.search(r"SEN\d+", str(text))
    return match.group(0) if match else None

# 4. 매핑 사전 생성 (SEN ID를 키로 하여 한글 라벨 저장)
# 예: {'SEN0001': '왼쪽', 'SEN0002': '왼쪽', ...}
meta_df['sen_key'] = meta_df['video_uid'].apply(extract_sen)
sen_to_label = dict(zip(meta_df['sen_key'], meta_df['label']))

# 5. 모델 인덱스(0, 1, 2...)와 한글 라벨 매핑 (최종 사전)
final_map = {int(i): sen_to_label.get(extract_sen(name), "알수없음") for i, name in enumerate(classes)}

# 6. JSON으로 저장
with open("final_label_map.json", "w", encoding="utf-8") as f:
    json.dump(final_map, f, indent=4, ensure_ascii=False)

print("성공! 'final_label_map.json' 파일이 생성되었습니다.")
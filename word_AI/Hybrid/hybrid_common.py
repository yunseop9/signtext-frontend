"""Hybrid Model specific configurations and data slicing utilities."""
from __future__ import annotations
import numpy as np

# 하이브리드 모델 전용 하이퍼파라미터 고정
HYBRID_SEED = 42
HYBRID_BATCH_SIZE = 256
HYBRID_EPOCHS = 80

# def extract_hands_and_remove_c(X: np.ndarray) -> np.ndarray:
#     """
#     원본 데이터(411D)에서 양손(126D)을 슬라이싱한 후, 
#     노이즈 원인인 신뢰도(c) 채널을 완벽하게 제거하여 84D(x, y)로 압축합니다.
#     """
#     # 1. 양손 데이터 추출 (MediaPipe 기준 포즈 이후 인덱스 75 ~ 200)
#     # 결과 Shape: (batch_size, 30, 126) -> 42개 점 * 3채널(x, y, c)
#     hands_X = X[:, :, 75:201]
    
#     # 2. 구조 변경 (x, y, c 채널을 분리하기 위해 Reshape)
#     batch_size = hands_X.shape[0]
#     seq_len = hands_X.shape[1]
#     num_points = 42  # 양손 랜드마크 점의 개수
    
#     # Shape: (batch_size, 30, 42, 3)
#     reshaped_X = hands_X.reshape((batch_size, seq_len, num_points, 3))
    
#     # 3. 신뢰도(c) 채널 버리기 (0번(x), 1번(y) 채널만 슬라이싱)
#     # Shape: (batch_size, 30, 42, 2)
#     xy_only = reshaped_X[:, :, :, :2]
    
#     # 4. 모델 입력용으로 다시 평탄화 (42 * 2 = 84D)
#     # 최종 Shape: (batch_size, 30, 84)
#     final_X = xy_only.reshape((batch_size, seq_len, 84))
    
#     return final_X


def extract_pose_hands_and_remove_c(X: np.ndarray) -> np.ndarray:
    """
    원본 데이터(411D)에서 포즈(상체)와 양손 데이터(201D)를 슬라이싱한 후, 
    노이즈 원인인 신뢰도(c) 채널을 제거하여 134D(x, y)로 압축합니다.
    """
    # 1. 포즈 + 양손 데이터 추출 (MediaPipe 기준 인덱스 0 ~ 200)
    # 구성: 포즈(75D) + 왼손(63D) + 오른손(63D) = 201D (총 67개 점 * 3채널)
    pose_hands_X = X[:, :, 0:201]
    
    # 2. 구조 변경 (x, y, c 채널 분리)
    batch_size = pose_hands_X.shape[0]
    seq_len = pose_hands_X.shape[1]
    num_points = 67  # 포즈(25) + 양손(42) = 67개 점
    
    # Shape: (batch_size, 30, 67, 3)
    reshaped_X = pose_hands_X.reshape((batch_size, seq_len, num_points, 3))
    
    # 3. 신뢰도(c) 채널 버리기 (0번(x), 1번(y) 채널만 슬라이싱)
    # Shape: (batch_size, 30, 67, 2)
    xy_only = reshaped_X[:, :, :, :2]
    
    # 4. 모델 입력용으로 다시 평탄화 (67 * 2 = 134D)
    # 최종 Shape: (batch_size, 30, 134)
    final_X = xy_only.reshape((batch_size, seq_len, 134))
    
    return final_X
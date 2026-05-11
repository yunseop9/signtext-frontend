import numpy as np
import matplotlib.pyplot as plt
import os

def find_hand_indices(file_path, top_n=126):
    """
    X.npy 데이터를 분석하여 변동성(Variance)이 큰 인덱스를 추출합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {os.path.abspath(file_path)}")

    print(f"데이터 로드 중: {file_path}...")
    X = np.load(file_path)
    print(f"로드 완료. Shape: {X.shape}")

    # 차원별 분산 계산 (시간축 기준)
    avg_variance = np.mean(np.var(X, axis=1), axis=0)
    
    # 시각화
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(avg_variance)), avg_variance, color='skyblue')
    plt.title("Feature Variance Analysis")
    plt.xlabel("Feature Index")
    plt.ylabel("Average Variance")
    plt.show()

    # 상위 분산 인덱스 추출
    top_indices = np.sort(np.argsort(avg_variance)[-top_n:])
    
    print("\n" + "="*50)
    print(f"분석 결과 (상위 {top_n}개)")
    print(f"인덱스 범위: {top_indices[0]} ~ {top_indices[-1]}")
    print("="*50)
    
    return top_indices

if __name__ == "__main__":
    # 1. 현재 스크립트의 절대 경로 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 제공해주신 구조에 따른 상대 경로 설정
    # 1D-CNN(0) -> word_AI(1) -> 99. 깃헙 코드(2) -> 9주차(3) -> 전처리 완료 데이터(단어)
    relative_path = os.path.join(current_dir, "..", "..", "..", "전처리 완료 데이터(단어)", "X.npy")
    
    try:
        hand_indices = find_hand_indices(relative_path)
        
        # 결과 저장 (현재 스크립트와 동일 폴더)
        save_path = os.path.join(current_dir, 'core_indices.txt')
        np.savetxt(save_path, hand_indices, fmt='%d')
        print(f"\n분석 완료! 인덱스가 다음 경로에 저장되었습니다:\n{save_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("경로 설정을 다시 확인해주세요.")
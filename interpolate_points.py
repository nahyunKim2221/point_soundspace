import numpy as np
from scipy.spatial import cKDTree

def interpolate_soundspaces_points(input_file, output_file, target_spacing=0.1, threshold=0.25):
    # 1. 데이터 로드 (index x y z)
    # 첫 번째 컬럼은 index이므로 제외하고 x, y, z만 추출
    data = np.loadtxt(input_file)
    indices = data[:, 0]
    original_coords = data[:, 1:]
    
    print(f"원본 포인트 개수: {len(original_coords)}")

    # 2. KD-Tree 구축 (근접 이웃 검색용)
    tree = cKDTree(original_coords)

    # 3. 전체 영역의 경계값 계산
    min_bound = original_coords.min(axis=0)
    max_bound = original_coords.max(axis=0)

    # 4. Y축(높이) 처리
    # 실내 환경은 층이 나뉘어 있을 수 있으므로, 원본에 존재하는 Y값들만 추출
    unique_heights = np.unique(original_coords[:, 1])
    
    new_points_list = []

    # 5. 각 층별로 0.1m 그리드 생성 및 필터링
    for h in unique_heights:
        x_range = np.arange(min_bound[0], max_bound[0] + target_spacing, target_spacing)
        z_range = np.arange(min_bound[2], max_bound[2] + target_spacing, target_spacing)
        
        xv, zv = np.meshgrid(x_range, z_range)
        # 해당 층(h)에서의 그리드 포인트들
        grid_candidates = np.stack([xv.ravel(), np.full_like(xv.ravel(), h), zv.ravel()], axis=-1)
        
        # 원본 데이터와의 거리 계산
        dists, _ = tree.query(grid_candidates)
        
        # 임계값(threshold) 이내에 있는 포인트만 유효한 Mesh 내부로 간주
        valid_points = grid_candidates[dists <= threshold]
        new_points_list.append(valid_points)

    # 모든 층의 포인트 합치기
    all_new_points = np.vstack(new_points_list)
    
    # 6. 중복 제거 (혹시 모를 중복 좌표 방지)
    all_new_points = np.unique(np.round(all_new_points, decimals=4), axis=0)

    # 7. 새로운 index 부여 및 저장
    final_data = []
    for i, pt in enumerate(all_new_points):
        # index x y z 형식
        final_data.append([float(i), pt[0], pt[1], pt[2]])
    
    final_data = np.array(final_data)
    
    # 결과 저장 (소수점 6자리까지)
    np.savetxt(output_file, final_data, fmt='%d %.6f %.6f %.6f')
    
    print(f"새로운 포인트 개수: {len(final_data)}")
    print(f"결과가 {output_file}에 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    # 파일 경로를 본인의 환경에 맞게 수정하세요.
    input_path = "points.txt" 
    output_path = "points_dense.txt"
    
    # threshold는 기존 간격(0.5m)의 절반보다 조금 큰 0.25~0.3m가 적당합니다.
    interpolate_soundspaces_points(input_path, output_path, target_spacing=0.1, threshold=0.3)

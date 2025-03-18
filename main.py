import cv2
import numpy as np
import os
import pytesseract

def detect_and_save_frames(image_path, output_dir):
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이미지 이진화
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 모든 사각형 프레임의 크기 수집
    frame_sizes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 너무 작은 프레임 제외
        if w > 30 and h > 30:
            frame_sizes.append((w, h))
    
    # 프레임 크기 분석
    sizes_array = np.array(frame_sizes)
    median_width = np.median(sizes_array[:, 0])
    median_height = np.median(sizes_array[:, 1])
    
    # 답안 프레임 크기 범위 설정 (중간값 기준 ±10% 허용)
    width_range = (median_width * 0.9, median_width * 1.1)
    height_range = (median_height * 0.9, median_height * 1.1)
    
    print(f"예상되는 답안 프레임 크기: {median_width:.0f} x {median_height:.0f}")
    print(f"허용 범위: 너비 {width_range[0]:.0f}~{width_range[1]:.0f}, 높이 {height_range[0]:.0f}~{height_range[1]:.0f}")
    
    # 이미지 중간점 계산
    height, width = image.shape[:2]
    mid_x = width // 2
    
    def process_side(side_image, side_thresh, is_right_side, offset_x=0):
        # 윤곽선 찾기
        contours, _ = cv2.findContours(side_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 답안 프레임 필터링
        valid_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 크기가 예상 범위 내인 프레임만 선택
            if (width_range[0] <= w <= width_range[1] and 
                height_range[0] <= h <= height_range[1]):
                valid_boxes.append((x + offset_x, y, w, h))
        
        # y 좌표로 행 그룹화
        row_groups = {}
        y_tolerance = median_height * 0.3  # 같은 행으로 인식할 y좌표 차이
        
        for box in valid_boxes:
            x, y, w, h = box
            assigned = False
            for row_y in row_groups.keys():
                if abs(y - row_y) < y_tolerance:
                    row_groups[row_y].append(box)
                    assigned = True
                    break
            if not assigned:
                row_groups[y] = [box]
        
        # 각 행 정렬 및 처리
        sorted_rows = []
        for row_y in sorted(row_groups.keys()):
            row_boxes = row_groups[row_y]
            # x 좌표로 정렬 (오른쪽에서 왼쪽으로)
            row_boxes.sort(key=lambda b: b[0], reverse=True)
            if len(row_boxes) >= 3:  # 3개 이상의 박스가 있는 행만 처리
                sorted_rows.append(row_boxes[:3])
        
        # 7행만 선택
        sorted_rows = sorted_rows[:7]
        
        # 프레임 추출 및 저장
        problem_num = 2 if is_right_side else 1
        
        for row_idx, row in enumerate(sorted_rows, 1):
            for col_idx, (x, y, w, h) in enumerate(row, 1):
                # 패딩 적용
                padding = 5
                x_adj = x - offset_x  # offset 조정
                frame = side_image[y+padding:y+h-padding, x_adj+padding:x_adj+w-padding]
                
                # 반복 회차 설정
                repeat_suffix = {1: "一回目", 2: "二回目", 3: "三回目"}[col_idx]
                
                # 파일 저장
                output_path = os.path.join(output_dir, f"frame_問{problem_num}_{row_idx}_{repeat_suffix}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"저장됨: {output_path}")
    
    # 왼쪽과 오른쪽 영역 처리
    left_thresh = thresh[:, :mid_x]
    right_thresh = thresh[:, mid_x:]
    process_side(image[:, :mid_x], left_thresh, False, 0)
    process_side(image[:, mid_x:], right_thresh, True, mid_x)
    
    print("프레임 추출 완료")

# 사용 예시
image_path = "writing_sample_3.jpg"  # 이미지 파일 경로
output_dir = "extracted_frames"  # 출력 디렉토리
detect_and_save_frames(image_path, output_dir)
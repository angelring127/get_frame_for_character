import cv2
import numpy as np
import os

def detect_and_save_frames(image_path, output_dir):
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 모든 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 프레임 크기 분석
    frame_sizes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 정사각형에 가까운 프레임만 선택
        if 0.8 < w/h < 1.2 and w > 30:
            frame_sizes.append((w, h))
    
    # 답안 프레임 크기 분석 (중간값 사용)
    sizes_array = np.array(frame_sizes)
    median_width = np.median(sizes_array[:, 0])
    median_height = np.median(sizes_array[:, 1])
    
    # 체크란 크기 범위 설정 (답안 프레임의 50% 기준, ±10% 허용)
    checkbox_size = median_width * 0.5
    checkbox_range = (int(checkbox_size * 0.9), int(checkbox_size * 1.1))
    
    print(f"\n=== 프레임 크기 분석 ===")
    print(f"답안 프레임 크기: {median_width:.0f} x {median_height:.0f}")
    print(f"체크란 크기 범위: {checkbox_range[0]}~{checkbox_range[1]}")
    print("=====================\n")
    
    # 체크란 감지
    height = image.shape[0]
    top_area = image[:height//4, :]  # 상단 25% 영역만 처리
    
    # 검은색 체크란 감지를 위한 이진화
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 50, 50])  # 더 어두운 검은색만 감지
    black_mask = cv2.inRange(top_area, lower_black, upper_black)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((2,2), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    
    # 디버깅을 위해 black_mask 저장
    cv2.imwrite('debug_black_mask.jpg', black_mask)
    
    # 체크란 윤곽선 찾기
    checkbox_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n감지된 체크란 후보 수: {len(checkbox_contours)}")
    
    checkboxes = []
    for i, cnt in enumerate(checkbox_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w/h if h != 0 else 0
        
        # 체크란 크기 조건 확인
        if (checkbox_range[0] <= w <= checkbox_range[1] and 
            checkbox_range[0] <= h <= checkbox_range[1] and 
            0.8 < aspect_ratio < 1.2):
            
            # 테두리를 제외한 내부 영역 추출 (20% 패딩)
            padding_ratio = 0.2
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
            
            # 박스 내부의 V 표시 확인
            roi = top_area[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, v_thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
            
            # 디버깅을 위해 ROI 저장
            cv2.imwrite(f'debug_checkbox_roi_{i}.jpg', roi)
            cv2.imwrite(f'debug_checkbox_thresh_{i}.jpg', v_thresh)
            
            white_pixels = np.sum(v_thresh == 255)
            total_pixels = (w-2*pad_x) * (h-2*pad_y)  # 내부 영역 크기
            white_ratio = white_pixels / total_pixels
            
            print(f"체크란 후보 {i}: 크기 {w}x{h}, 비율 {aspect_ratio:.2f}, 흰색 비율 {white_ratio:.3f}")
            
            checkboxes.append({
                'position': (x, y),
                'size': (w, h),
                'white_ratio': white_ratio,
                'is_checked': False,
                'number': None  # 문제 번호를 저장할 필드 추가
            })
    
    # 체크박스들을 x 좌표 기준으로 정렬
    checkboxes.sort(key=lambda box: box['position'][0], reverse=True)  # x 좌표 큰 순서대로 정렬
    
    # 체크박스 번호 할당 (영문으로 변경)
    if len(checkboxes) >= 4:
        checkbox_numbers = ['question03', 'question01', 'question04', 'question02']
        for i, box in enumerate(checkboxes[:4]):
            box['number'] = checkbox_numbers[i]
    
    # 첫 번째 문제 체크 (checkbox 1, 2 비교)
    if len(checkboxes) >= 2:
        box1, box2 = checkboxes[0], checkboxes[1]  # question03, question01
        if box1['white_ratio'] < box2['white_ratio']:
            box1['is_checked'] = True
        elif box2['white_ratio'] < box1['white_ratio']:
            box2['is_checked'] = True
        else:  # 동일하거나 체크가 없는 경우
            box2['is_checked'] = True  # question01을 기본값으로 설정
    
    # 두 번째 문제 체크 (checkbox 3, 4 비교)
    if len(checkboxes) >= 4:
        box3, box4 = checkboxes[2], checkboxes[3]  # question04, question02
        if box3['white_ratio'] < box4['white_ratio']:
            box3['is_checked'] = True
        elif box4['white_ratio'] < box3['white_ratio']:
            box4['is_checked'] = True
        else:  # 동일하거나 체크가 없는 경우
            box3['is_checked'] = True  # question04를 기본값으로 설정
    
    # 체크란 표시 및 결과 출력
    print("\n=== 체크란 감지 결과 ===")
    for box in checkboxes:
        x, y = box['position']
        w, h = box['size']
        color = (0, 255, 0) if box['is_checked'] else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        status = "체크됨 ✓" if box['is_checked'] else "체크되지 않음 ✗"
        print(f"{box['number']}: {status} (흰색 비율: {box['white_ratio']:.3f})")
    
    print("\n=== 문제별 선택된 답안 ===")
    selected_q1 = next((box['number'] for box in checkboxes[:2] if box['is_checked']), 'question01')
    selected_q2 = next((box['number'] for box in checkboxes[2:4] if box['is_checked']), 'question04')
    print(f"문제1: {selected_q1}")
    print(f"문제2: {selected_q2}")
    print("=====================\n")
    
    # 디버깅을 위해 처리된 이미지 저장
    cv2.imwrite('debug_checkboxes.jpg', image)
    
    # 체크란 이미지 저장
    for i, box in enumerate(checkboxes, 1):
        x, y = box['position']
        w, h = box['size']
        padding = 2
        checkbox = image[y-padding:y+h+padding, x-padding:x+w+padding]
        output_path = os.path.join(output_dir, f"checkbox_{i}.jpg")
        cv2.imwrite(output_path, checkbox)
        print(f"체크란 저장됨: {output_path}")
    
    # 답안 프레임 필터링 및 저장
    valid_frames = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 답안 프레임 크기 범위 내의 것만 선택
        if (median_width * 0.9 <= w <= median_width * 1.1 and 
            median_height * 0.9 <= h <= median_height * 1.1):
            valid_frames.append((x, y, w, h))
    
    # y 좌표로 행 그룹화
    row_groups = {}
    y_tolerance = median_height * 0.3  # 같은 행으로 인식할 y좌표 차이
    
    for frame in valid_frames:
        x, y, w, h = frame
        assigned = False
        for row_y in row_groups.keys():
            if abs(y - row_y) < y_tolerance:
                row_groups[row_y].append(frame)
                assigned = True
                break
        if not assigned:
            row_groups[y] = [frame]
    
    # 이미지 중간점 계산
    width = image.shape[1]
    mid_x = width // 2
    
    # 각 행 정렬 및 처리
    sorted_rows = []
    for row_y in sorted(row_groups.keys()):
        row_frames = row_groups[row_y]
        # x 좌표로 정렬
        row_frames.sort(key=lambda f: f[0])
        sorted_rows.append(row_frames)
    
    # 7행만 선택하고 각 행의 프레임을 왼쪽/오른쪽으로 분류
    sorted_rows = sorted_rows[:7]
    for row_idx, row in enumerate(sorted_rows, 1):
        left_frames = [f for f in row if f[0] < mid_x]
        right_frames = [f for f in row if f[0] >= mid_x]
        
        # 왼쪽 프레임 처리 (문제2)
        for col_idx, (x, y, w, h) in enumerate(sorted(left_frames, key=lambda f: f[0]), 1):
            if col_idx > 3: continue
            padding = 5
            frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
            repeat_suffix = {1: "03", 2: "02", 3: "01"}[col_idx]  # 역순으로 변경
            output_path = os.path.join(output_dir, f"frame_{selected_q2}_{row_idx}_{repeat_suffix}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path}")
        
        # 오른쪽 프레임 처리 (문제1)
        for col_idx, (x, y, w, h) in enumerate(sorted(right_frames, key=lambda f: f[0]), 1):
            if col_idx > 3: continue
            padding = 5
            frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
            repeat_suffix = {1: "03", 2: "02", 3: "01"}[col_idx]  # 역순으로 변경
            output_path = os.path.join(output_dir, f"frame_{selected_q1}_{row_idx}_{repeat_suffix}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path}")
    
    print("\n프레임 추출 완료")

# 사용 예시
image_path = "writing_sample_3.jpg"  # 이미지 파일 경로
output_dir = "extracted_frames"  # 출력 디렉토리
detect_and_save_frames(image_path, output_dir)
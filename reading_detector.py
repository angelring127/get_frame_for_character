import cv2
import numpy as np
import os

def detect_checkboxes(image, avg_width, output_dir):
    """
    이미지에서 체크란을 감지하고 선택된 문제 번호를 반환합니다.
    """
    height = image.shape[0]
    width = image.shape[1]
    right_area = image[:, int(width*0.82):]  # 오른쪽 18% 영역만 처리
    
    # 체크란 크기 범위 설정 (답안 프레임의 50% 기준, ±15% 허용)
    checkbox_size = avg_width * 0.5
    checkbox_range = (int(checkbox_size * 0.85), int(checkbox_size * 1.15))
    
    print(f"체크란 크기 범위: {checkbox_range[0]}~{checkbox_range[1]}")
    
    # 검은색 체크란 감지를 위한 이진화 (임계값 조정)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([80, 80, 80])  # 임계값 증가
    black_mask = cv2.inRange(right_area, lower_black, upper_black)
    
    # 모폴로지 연산으로 노이즈 제거 및 체크란 강화
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    
    # 디버깅을 위해 black_mask 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_black_mask.jpg'), black_mask)
    
    # 체크란 윤곽선 찾기
    checkbox_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n감지된 체크란 후보 수: {len(checkbox_contours)}")
    
    checkboxes = []
    for i, cnt in enumerate(checkbox_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w/h if h != 0 else 0
        
        # 체크란 크기 조건 확인 (조건 완화)
        if (checkbox_range[0] <= w <= checkbox_range[1] and 
            checkbox_range[0] <= h <= checkbox_range[1] and 
            0.75 < aspect_ratio < 1.25):  # 비율 범위 확대
            
            # 테두리를 제외한 내부 영역 추출 (패딩 축소)
            padding_ratio = 0.05  # 패딩 비율 축소
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
            
            # 박스 내부 영역 추출 및 적응형 이진화 적용
            roi = right_area[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]
            if roi.size > 0:  # ROI가 유효한 경우에만 처리
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(
                    gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                
                white_pixels = np.sum(thresh == 255)
                total_pixels = roi.shape[0] * roi.shape[1]
                white_ratio = white_pixels / total_pixels
                
                print(f"체크란 후보 {i}: 크기 {w}x{h}, 비율 {aspect_ratio:.2f}, 흰색 비율 {white_ratio:.3f}")
                
                checkboxes.append({
                    'position': (x + int(width*0.82), y),  # 전체 이미지 기준 좌표로 변환 (0.82로 수정)
                    'size': (w, h),
                    'white_ratio': white_ratio,
                    'is_checked': False,
                    'number': None
                })
    
    # 체크박스들을 x 좌표가 비슷한 것들끼리 그룹화
    if len(checkboxes) >= 4:
        # x 좌표 기준으로 정렬
        checkboxes.sort(key=lambda box: box['position'][0])
        
        # x 좌표가 비슷한 체크박스들을 그룹화 (허용 오차: avg_width의 10%)
        x_tolerance = avg_width * 0.1
        grouped_boxes = []
        current_group = [checkboxes[0]]
        
        for box in checkboxes[1:]:
            if abs(box['position'][0] - current_group[0]['position'][0]) <= x_tolerance:
                current_group.append(box)
            else:
                if len(current_group) >= 4:  # 유효한 그룹만 저장
                    grouped_boxes.extend(current_group[:4])
                current_group = [box]
        
        if len(current_group) >= 4:  # 마지막 그룹 처리
            grouped_boxes.extend(current_group[:4])
        
        # y 좌표로 정렬하여 1-4번 할당
        grouped_boxes.sort(key=lambda box: box['position'][1])
        checkboxes = grouped_boxes[:4]
        
        for i, box in enumerate(checkboxes):
            box['number'] = f"question{i+1}"
    
    # 체크박스 감지 부분 수정
    selected_question = 'question01'  # 기본값 설정
    
    if len(checkboxes) >= 4:
        # 흰색 비율이 가장 낮은 체크박스 찾기 (임계값 조정)
        min_white_ratio = float('inf')
        selected_box = None
        
        for box in checkboxes[:4]:
            # 흰색 비율이 0.85 미만이고 가장 낮은 경우를 체크된 것으로 간주
            if box['white_ratio'] < 0.85 and box['white_ratio'] < min_white_ratio:
                min_white_ratio = box['white_ratio']
                selected_box = box
        
        # 선택된 체크박스 표시
        if selected_box:
            selected_box['is_checked'] = True
            selected_question = selected_box['number']
        else:
            # 체크된 것이 없는 경우 기본값 설정
            checkboxes[0]['is_checked'] = True
            selected_question = 'question01'
    
    # 체크란 표시 및 결과 출력
    print("\n=== 체크란 감지 결과 ===")
    for box in checkboxes:
        x, y = box['position']
        w, h = box['size']
        color = (0, 255, 0) if box['is_checked'] else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        status = "체크됨 ✓" if box['is_checked'] else "체크되지 않음 ✗"
        print(f"{box['number']}: {status} (흰색 비율: {box['white_ratio']:.3f})")
    
    print("\n=== 선택된 답안 ===")
    print(f"선택된 문제: {selected_question}")
    print("=====================\n")
    
    # 디버깅을 위해 처리된 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_checkboxes.jpg'), image)
    
    return selected_question, checkboxes

def clean_frame_border(frame, is_edge=False):
    """
    프레임 이미지의 테두리 잔여물을 제거합니다.
    가장자리 프레임의 경우 더 강력한 처리를 적용합니다.
    """
    # 원본 이미지 보존
    original = frame.copy()
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거를 위한 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 가장자리 프레임의 경우 더 강력한 처리
    if is_edge:
        # 더 강력한 블러 적용
        blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
        
        # 적응형 이진화 임계값 조정
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
    else:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    # 모폴로지 연산으로 테두리 정리
    kernel_size = 5 if is_edge else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 찾기
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 여백 설정 (가장자리 프레임은 더 큰 여백)
        padding = 8 if is_edge else 5
        x = max(0, x + padding)
        y = max(0, y + padding)
        w = min(frame.shape[1] - x, w - 2*padding)
        h = min(frame.shape[0] - y, h - 2*padding)
        
        # 잘라낸 영역이 유효한지 확인
        if w > 0 and h > 0:
            # 내부 영역 추출
            cleaned_frame = original[y:y+h, x:x+w]
            
            # 테두리 블렌딩
            border_width = 3 if is_edge else 2
            alpha = 0.2 if is_edge else 0.1
            
            # 흰색 배경과 블렌딩
            white_border = np.ones_like(cleaned_frame) * 255
            
            # 테두리 블렌딩 적용
            cleaned_frame[:border_width] = cv2.addWeighted(
                cleaned_frame[:border_width], 1-alpha,
                white_border[:border_width], alpha, 0
            )
            cleaned_frame[-border_width:] = cv2.addWeighted(
                cleaned_frame[-border_width:], 1-alpha,
                white_border[-border_width:], alpha, 0
            )
            cleaned_frame[:, :border_width] = cv2.addWeighted(
                cleaned_frame[:, :border_width], 1-alpha,
                white_border[:, :border_width], alpha, 0
            )
            cleaned_frame[:, -border_width:] = cv2.addWeighted(
                cleaned_frame[:, -border_width:], 1-alpha,
                white_border[:, -border_width:], alpha, 0
            )
            
            return cleaned_frame
    
    return frame

def create_a4_canvas(image_width, image_height, avg_width, avg_height, avg_col_spacing, avg_row_spacing):
    """
    A4 크기의 흰색 캔버스를 생성합니다.
    A4 크기는 210mm x 297mm이며, 300dpi에서는 2480 x 3508 픽셀입니다.
    """
    # A4 크기 (300dpi 기준)
    a4_width = 2480
    a4_height = 3508
    
    # 여백 설정 (상하좌우 각각 5% 여백)
    margin_ratio = 0.05
    usable_width = int(a4_width * (1 - 2 * margin_ratio))
    usable_height = int(a4_height * (1 - 2 * margin_ratio))
    
    # 원본 이미지의 비율을 유지하면서 A4 크기에 맞게 조정
    scale_factor = min(usable_width / image_width, usable_height / image_height)
    
    # 스케일링된 이미지 크기
    scaled_width = int(image_width * scale_factor)
    scaled_height = int(image_height * scale_factor)
    
    # 중앙 정렬을 위한 여백 계산
    left_margin = int((a4_width - scaled_width) / 2)
    top_margin = int((a4_height - scaled_height) / 2)
    
    # 흰색 배경 생성
    canvas = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255
    
    return canvas, scale_factor, left_margin, top_margin

def merge_images_on_canvas(image_path, output_dir, canvas, scale_factor, left_margin, top_margin, frame_grid, avg_width, avg_height, avg_col_spacing, avg_row_spacing):
    """
    추출된 이미지들을 A4 캔버스에 원래 위치 그대로 배치합니다.
    """
    # 원본 이미지 로드
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 각 열의 x 좌표와 각 행의 y 좌표 수집
    col_x_coords = [[] for _ in range(7)]
    row_y_coords = [[] for _ in range(12)]
    
    for row in range(12):
        for col in range(7):
            frame = frame_grid[row][col]
            if frame is not None:
                x, y, w, h = frame
                col_x_coords[col].append(x)
                row_y_coords[row].append(y)
    
    # 각 열과 행의 대표 좌표 계산 (중간값 사용)
    col_positions = []
    row_positions = []
    
    for col_coords in col_x_coords:
        if col_coords:
            col_positions.append(int(np.median(col_coords)))
        else:
            if col_positions:
                col_positions.append(col_positions[-1] + int(avg_col_spacing))
            else:
                col_positions.append(0)
    
    for row_coords in row_y_coords:
        if row_coords:
            row_positions.append(int(np.median(row_coords)))
        else:
            if row_positions:
                row_positions.append(row_positions[-1] + int(avg_row_spacing))
            else:
                row_positions.append(0)
    
    # 최소 좌표 계산
    min_x = min(col_positions)
    min_y = min(row_positions)
    
    # 캔버스에 이미지 배치
    for row in range(12):
        for col in range(7):
            frame = frame_grid[row][col]
            if frame is not None:
                x, y, w, h = frame
                
                # 원본 이미지에서 프레임 추출
                padding = 5
                frame_img = original_image[y+padding:y+h-padding, x+padding:x+w-padding]
                
                # 가장자리 프레임 여부 확인
                is_edge = (col == 0 or col == 6 or row == 11)
                frame_img = clean_frame_border(frame_img, is_edge)
                
                if frame_img is not None and frame_img.size > 0:
                    # 스케일 조정
                    new_w = int(frame_img.shape[1] * scale_factor)
                    new_h = int(frame_img.shape[0] * scale_factor)
                    frame_img = cv2.resize(frame_img, (new_w, new_h))
                    
                    # 정렬된 위치 계산 (스케일 적용 + 여백 추가)
                    canvas_x = left_margin + int((col_positions[col] - min_x) * scale_factor)
                    canvas_y = top_margin + int((row_positions[row] - min_y) * scale_factor)
                    
                    # 캔버스 범위 확인
                    if (canvas_y + new_h <= canvas.shape[0] and 
                        canvas_x + new_w <= canvas.shape[1]):
                        try:
                            canvas[canvas_y:canvas_y+new_h, 
                                  canvas_x:canvas_x+new_w] = frame_img
                        except ValueError as e:
                            print(f"이미지 배치 중 오류 발생: {e}")
                            print(f"위치: ({row}, {col}), 크기: {frame_img.shape}")
    
    # 결과 저장
    output_path = os.path.join(output_dir, 'merged_a4.jpg')
    cv2.imwrite(output_path, canvas)
    print(f"A4 이미지 병합 완료: {output_path}")

def reading_detect_and_save_frames(image_path, output_dir):
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 이미지 전처리 강화
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 노이즈 제거를 위한 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 적응형 이진화 적용
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 모폴로지 연산으로 노이즈 제거 및 윤곽선 강화
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 디버깅을 위해 전처리된 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_preprocessed.jpg'), thresh)
    
    # 모든 윤곽선 찾기 (RETR_EXTERNAL 대신 RETR_LIST 사용)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 답안 프레임 필터링 및 저장
    all_frames = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 프레임 크기 조건 완화
        if 0.7 < w/h < 1.3 and w > 20:  # 정사각형 비율 범위 확대, 최소 크기 감소
            all_frames.append((x, y, w, h))
    
    print(f"\n감지된 모든 프레임 수: {len(all_frames)}")
    
    # 평균 프레임 크기 계산 (이상치 제거)
    frame_sizes = [(w, h) for _, _, w, h in all_frames]
    sizes_array = np.array(frame_sizes)
    
    # IQR 방식으로 이상치 제거
    q1_width = np.percentile(sizes_array[:, 0], 25)
    q3_width = np.percentile(sizes_array[:, 0], 75)
    iqr_width = q3_width - q1_width
    lower_bound_width = q1_width - 1.5 * iqr_width
    upper_bound_width = q3_width + 1.5 * iqr_width
    
    q1_height = np.percentile(sizes_array[:, 1], 25)
    q3_height = np.percentile(sizes_array[:, 1], 75)
    iqr_height = q3_height - q1_height
    lower_bound_height = q1_height - 1.5 * iqr_height
    upper_bound_height = q3_height + 1.5 * iqr_height
    
    # 이상치를 제외한 프레임만 선택
    valid_sizes = sizes_array[
        (sizes_array[:, 0] >= lower_bound_width) & 
        (sizes_array[:, 0] <= upper_bound_width) &
        (sizes_array[:, 1] >= lower_bound_height) & 
        (sizes_array[:, 1] <= upper_bound_height)
    ]
    
    avg_width = np.median(valid_sizes[:, 0])
    avg_height = np.median(valid_sizes[:, 1])
    
    print(f"\n=== 프레임 크기 분석 ===")
    print(f"평균 프레임 크기: {avg_width:.0f} x {avg_height:.0f}")
    print(f"이상치 제거 전 프레임 수: {len(sizes_array)}")
    print(f"이상치 제거 후 프레임 수: {len(valid_sizes)}")
    print("=====================\n")
    
    # 체크란 감지
    selected_question, checkboxes = detect_checkboxes(image, avg_width, output_dir)
    
    # 선택된 문제 번호 추출
    selected_number = selected_question.replace('question', '')
    
    # 체크란 이미지 저장
    for i, box in enumerate(checkboxes, 1):
        x, y = box['position']
        w, h = box['size']
        padding = 2
        checkbox = image[y-padding:y+h+padding, x-padding:x+w-padding]
        # 테두리 잔여물 제거
        checkbox = clean_frame_border(checkbox)
        output_path = os.path.join(output_dir, f"answer{selected_number}_checkbox_{i}.jpg")
        cv2.imwrite(output_path, checkbox)
        print(f"체크란 저장됨: {output_path}")
    
    # 유효한 프레임 범위 설정 (±15%)
    valid_width_range = (avg_width * 0.85, avg_width * 1.15)
    valid_height_range = (avg_height * 0.85, avg_height * 1.15)
    
    # 유효한 프레임만 선택
    valid_frames = []
    for x, y, w, h in all_frames:
        if (valid_width_range[0] <= w <= valid_width_range[1] and 
            valid_height_range[0] <= h <= valid_height_range[1]):
            valid_frames.append((x, y, w, h))
    
    print(f"유효한 프레임 수: {len(valid_frames)}")
    
    # x, y 좌표 분석
    x_coords = sorted([f[0] for f in valid_frames])
    y_coords = sorted([f[1] for f in valid_frames])
    
    # 열 간격 계산 (x 좌표 차이의 중간값)
    x_diffs = []
    for i in range(len(x_coords)-1):
        diff = x_coords[i+1] - x_coords[i]
        if diff > avg_width * 0.5:  # 최소 간격 설정
            x_diffs.append(diff)
    avg_col_spacing = np.median(x_diffs) if x_diffs else avg_width * 1.2
    
    # 행 간격 계산 (y 좌표 차이의 중간값)
    y_diffs = []
    for i in range(len(y_coords)-1):
        diff = y_coords[i+1] - y_coords[i]
        if diff > avg_height * 0.5:  # 최소 간격 설정
            y_diffs.append(diff)
    avg_row_spacing = np.median(y_diffs) if y_diffs else avg_height * 1.2
    
    print(f"\n=== 프레임 간격 분석 ===")
    print(f"평균 열 간격: {avg_col_spacing:.1f}픽셀")
    print(f"평균 행 간격: {avg_row_spacing:.1f}픽셀")
    print("=====================\n")
    
    # 12x7 크기의 2차원 배열 초기화
    frame_grid = [[None for _ in range(7)] for _ in range(12)]
    
    # 각 프레임의 위치를 행과 열 인덱스로 변환
    for x, y, w, h in valid_frames:
        # 열 인덱스 계산 (왼쪽에서 오른쪽으로)
        if avg_col_spacing > 0:
            col_idx = round((x - x_coords[0]) / avg_col_spacing)
            col_idx = max(0, min(6, col_idx))  # 0-6 범위로 제한
            
            # 행 인덱스 계산 (위에서 아래로)
            if avg_row_spacing > 0:
                row_idx = round((y - y_coords[0]) / avg_row_spacing)
                row_idx = max(0, min(11, row_idx))  # 0-11 범위로 제한
                
                frame_grid[row_idx][col_idx] = (x, y, w, h)
    
    # 각 행의 프레임 수 확인 및 출력
    print("\n=== 행별 프레임 수 ===")
    for i, row in enumerate(frame_grid):
        valid_count = sum(1 for f in row if f is not None)
        print(f"행 {i}: {valid_count}개 프레임")
    print("=====================\n")
    
    # 추출된 프레임 위치 저장을 위한 집합
    extracted_positions = set()
    
    # 각 question(1-7)에 대한 행-열 매핑 처리
    total_frames = 0
    for col in range(7):  # 7개 열 (0-6)
        question_num = 7 - col  # 7부터 1까지 역순
        
        for row in range(12):  # 12개 행 (0-11)
            frame = frame_grid[row][col]
            if frame is None:
                # 누락된 프레임의 예상 위치 계산
                if row > 0 and frame_grid[row-1][col] is not None:
                    prev_x, prev_y, prev_w, prev_h = frame_grid[row-1][col]
                    expected_x = prev_x
                    expected_y = prev_y + avg_row_spacing
                    
                    # 예상 위치 주변에서 프레임 검색
                    search_area = image[
                        max(0, int(expected_y - avg_row_spacing/2)):min(image.shape[0], int(expected_y + avg_row_spacing/2)),
                        max(0, int(expected_x - prev_w)):min(image.shape[1], int(expected_x + prev_w))
                    ]
                    
                    if search_area.size > 0:
                        # 검색 영역에서 프레임 크기의 윤곽선 찾기
                        gray_search = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
                        _, thresh_search = cv2.threshold(gray_search, 150, 255, cv2.THRESH_BINARY_INV)
                        search_contours, _ = cv2.findContours(thresh_search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in search_contours:
                            sx, sy, sw, sh = cv2.boundingRect(cnt)
                            if (valid_width_range[0] <= sw <= valid_width_range[1] and 
                                valid_height_range[0] <= sh <= valid_height_range[1]):
                                # 프레임 찾음
                                frame = search_area[sy:sy+sh, sx:sx+sw]
                                # 테두리 잔여물 제거
                                frame = clean_frame_border(frame)
                                
                                # 파일명 생성 (모든 프레임에 answer 접두사 추가)
                                output_path = os.path.join(
                                    output_dir, 
                                    f"answer{selected_number}_question{question_num}_repeat{(row//4)+1}_box{(row%4)+1}.jpg"
                                )
                                cv2.imwrite(output_path, frame)
                                print(f"추가 저장됨: {output_path} (위치: {col},{row})")
                                total_frames += 1
                                break
                    else:
                        print(f"경고: question{question_num}_repeat{(row//4)+1}_box{(row%4)+1} 누락됨 (위치: {col},{row})")
                continue
            
            x, y, w, h = frame
            # 중복 방지를 위한 위치 체크
            position_key = (x, y)
            if position_key not in extracted_positions:
                padding = 5
                frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
                # 테두리 잔여물 제거
                frame = clean_frame_border(frame)
                
                # repeat와 box 번호 계산
                repeat_num = (row // 4) + 1  # 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
                box_num = (row % 4) + 1      # 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4
                
                # 파일명 생성 (모든 프레임에 answer 접두사 추가)
                output_path = os.path.join(
                    output_dir, 
                    f"answer{selected_number}_question{question_num}_repeat{repeat_num}_box{box_num}.jpg"
                )
                cv2.imwrite(output_path, frame)
                print(f"저장됨: {output_path} (위치: {col},{row})")
                extracted_positions.add(position_key)
                total_frames += 1
    
    print(f"\n총 추출된 프레임 수: {total_frames}/84")
    if total_frames < 84:
        print("경고: 일부 프레임이 누락되었습니다.")
    print("\n프레임 추출 완료")
    
    # A4 캔버스 생성 및 이미지 병합
    canvas, scale_factor, left_margin, top_margin = create_a4_canvas(
        image.shape[1], image.shape[0], 
        avg_width, avg_height, 
        avg_col_spacing, avg_row_spacing
    )
    
    # 캔버스에 이미지 병합
    merge_images_on_canvas(
        image_path, output_dir, canvas, scale_factor, 
        left_margin, top_margin, frame_grid, 
        avg_width, avg_height, 
        avg_col_spacing, avg_row_spacing
    ) 
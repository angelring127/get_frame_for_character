import cv2
import numpy as np
import os

def clean_frame_border(frame):
    """
    프레임 이미지의 테두리 잔여물을 제거합니다.
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 이진화
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산으로 테두리 정리
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 찾기
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 여백 추가 (내부 영역 보존)
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # 정리된 영역 추출
        cleaned_frame = frame[y:y+h, x:x+w]
        return cleaned_frame
    
    return frame

def create_template_output(image, template_path, output_dir, template_name):
    """
    템플릿 이미지에 프레임을 배치하여 출력합니다.
    각 문제의 21개 이미지를 해당 문제의 시작 위치부터 순차적으로 배치합니다.
    템플릿 이미지는 5000x5000px 크기로 고정됩니다.
    프레임 크기: 130x130px, 간격: 20px
    """
    # 템플릿 이미지 로드
    template = cv2.imread(template_path)
    if template is None:
        # 템플릿 이미지를 찾을 수 없는 경우 output 디렉토리에서 찾기 시도
        output_template_path = os.path.join("output", os.path.basename(template_path))
        template = cv2.imread(output_template_path)
        
        if template is None:
            # output 디렉토리에서도 찾을 수 없는 경우 기본 템플릿 사용
            if os.path.basename(template_path) != "template.png":
                template = cv2.imread("template.png")
                if template is None:
                    raise ValueError(f"템플릿 이미지를 불러올 수 없습니다: {template_path}, {output_template_path}, template.png")
            else:
                raise ValueError(f"템플릿 이미지를 불러올 수 없습니다: {template_path}")
    
    # 템플릿 크기를 5000x5000으로 고정
    template = cv2.resize(template, (5000, 5000))
    output = template.copy()
    
    # 템플릿 프레임 설정
    frame_size = 130
    rows = 23
    cols = 22
    
    # 시작 위치 계산 (중앙 정렬)
    start_x = (template.shape[1] - (cols * (frame_size))) // 2
    start_y = (template.shape[0] - (rows * (frame_size))) // 2
    
    # 추출된 이미지 파일 목록 가져오기
    extracted_files = []
    for file in os.listdir(output_dir):
        if file.startswith('frame_question') and file.endswith('.jpg'):
            # 파일명에서 번호 추출 (frame_question02_1_03 형식)
            parts = file.replace('.jpg', '').split('_')
            question = parts[1].replace('question', '')  # question02 -> 02
            # 두 자리 숫자로 맞추기
            if len(question) == 1:
                question = f"0{question}"
            repeat = int(parts[2])  # 1
            box = int(parts[3])     # 03
            extracted_files.append((file, question, repeat, box))
    
    # 파일 정렬: question > repeat > box 순서
    extracted_files.sort(key=lambda x: (x[1], x[2], x[3]))
    
    if not extracted_files:
        raise ValueError("프레임 이미지를 찾을 수 없습니다.")
    
    # 각 question 번호별로 파일 그룹화
    question_files = {}
    for file_info in extracted_files:
        question = file_info[1]
        if question not in question_files:
            question_files[question] = []
        question_files[question].append(file_info)
    
    # 문제 번호별 시작 위치 계산
    question_start_positions = {
        "01": (1, 0),    # 1번 문제: 1번째 줄, 1번째 칸부터
        "02": (1, 21),   # 2번 문제: 1번째 줄, 22번째 칸부터
        "03": (2, 20),   # 3번 문제: 2번째 줄, 21번째 칸부터
        "04": (3, 19),   # 4번 문제: 3번째 줄, 20번째 칸부터
    }
    
    # 각 question 번호별로 프레임 배치 (01부터 04까지 순서대로)
    for question_num in sorted(question_files.keys()):
        if question_num not in question_start_positions:
            print(f"경고: 알 수 없는 문제 번호 {question_num}는 건너뜁니다.")
            continue
            
        files = question_files[question_num]
        # 시작 위치 설정
        start_row, start_col = question_start_positions[question_num]
        print(f"\n문제 {question_num} 프레임 배치 시작 [시작 위치: 행={start_row}, 열={start_col}]")
        
        # 프레임 위치 계산 (21개의 프레임을 시작 위치부터 순차적으로)
        template_frames = []
        frame_count = 0
        current_row = start_row
        current_col = start_col
        
        while frame_count < 21 and current_row < rows:
            x = start_x + current_col * (frame_size)
            y = start_y + current_row * (frame_size)
            template_frames.append((x, y, frame_size, frame_size))
            frame_count += 1
            
            # 다음 위치 계산
            current_col += 1
            if current_col >= cols:  # 열이 끝나면 다음 행으로
                current_col = 0
                current_row += 1
        
        # 프레임 배치
        for i, file_info in enumerate(files):
            if i >= len(template_frames):
                break
                
            file_name = file_info[0]
            frame_path = os.path.join(output_dir, file_name)
            
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"프레임을 로드할 수 없습니다: {frame_path}")
                continue
            
            # 템플릿의 현재 프레임 위치
            tx, ty, tw, th = template_frames[i]
            
            # 프레임 크기 조정 (비율 유지)
            target_size = (tw-4, th-4)  # 여백을 위해 약간 작게
            frame_ratio = frame.shape[1] / frame.shape[0]
            target_ratio = target_size[0] / target_size[1]
            
            if frame_ratio > target_ratio:
                new_width = target_size[0]
                new_height = int(new_width / frame_ratio)
            else:
                new_height = target_size[1]
                new_width = int(new_height * frame_ratio)
            
            frame = cv2.resize(frame, (new_width, new_height))
            
            # 중앙 정렬을 위한 오프셋 계산
            x_offset = tx + 2 + (tw - new_width) // 2
            y_offset = ty + 2 + (th - new_height) // 2
            
            try:
                output[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
                print(f"프레임 배치: {file_name} -> ({x_offset}, {y_offset}) [행:{current_row}, 열:{current_col}]")
            except ValueError as e:
                print(f"프레임 배치 오류: {e}")
                continue
    
    # 결과 이미지 저장 (output 디렉토리에 저장)
    output_path = os.path.join("output", f"{template_name}.jpg")
    
    # output 디렉토리가 없으면 생성
    if not os.path.exists("output"):
        os.makedirs("output")
        
    cv2.imwrite(output_path, output)
    print(f"템플릿 출력 이미지 저장됨: {output_path}")
    return output

def writing_detect_and_save_frames(image, output_dir, template_path=None, template_name=None):
    """
    전처리된 이미지에서 프레임을 감지하고 저장합니다.
    
    Args:
        image: 전처리된 이미지 (numpy.ndarray)
        output_dir: 출력 디렉토리 경로
        template_path: 템플릿 이미지 경로 (선택사항)
        template_name: 템플릿 이름 (선택사항)
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화 적용
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # 수직선과 수평선 감지를 위한 구조화 요소
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # 수직선 감지
    vertical_lines = cv2.erode(binary, vertical_kernel)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)
    
    # 수평선 감지
    horizontal_lines = cv2.erode(binary, horizontal_kernel)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)
    
    # 격자 구조 결합
    grid = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)
    
    # 디버깅을 위해 전처리된 이미지들 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_binary.jpg'), binary)
    cv2.imwrite(os.path.join(output_dir, 'debug_vertical.jpg'), vertical_lines)
    cv2.imwrite(os.path.join(output_dir, 'debug_horizontal.jpg'), horizontal_lines)
    cv2.imwrite(os.path.join(output_dir, 'debug_grid.jpg'), grid)
    
    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(
        grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 답안 프레임 필터링
    all_frames = []
    min_area = image.shape[0] * image.shape[1] * 0.001  # 최소 영역 크기
    max_area = image.shape[0] * image.shape[1] * 0.02   # 최대 영역 크기
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w/h if h != 0 else 0
            
            # 정사각형에 가까운 프레임만 선택
            if 0.8 < aspect_ratio < 1.2:
                all_frames.append((x, y, w, h))
    
    print(f"\n감지된 모든 프레임 수: {len(all_frames)}")
    
    # 디버깅을 위해 감지된 프레임 시각화
    debug_frame = image.copy()
    for x, y, w, h in all_frames:
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, 'debug_frames.jpg'), debug_frame)
    
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
    
    # 체크란 크기 범위 설정 (답안 프레임의 50% 기준, ±10% 허용)
    checkbox_size = avg_width * 0.5
    checkbox_range = (int(checkbox_size * 0.9), int(checkbox_size * 1.1))
    
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
    cv2.imwrite(os.path.join(output_dir, 'debug_black_mask.jpg'), black_mask)
    
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
            cv2.imwrite(os.path.join(output_dir, f'debug_checkbox_roi_{i}.jpg'), roi)
            cv2.imwrite(os.path.join(output_dir, f'debug_checkbox_thresh_{i}.jpg'), v_thresh)
            
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
    cv2.imwrite(os.path.join(output_dir, 'debug_checkboxes.jpg'), image)
    
    # 체크란 이미지 저장
    for i, box in enumerate(checkboxes, 1):
        x, y = box['position']
        w, h = box['size']
        padding = 2
        checkbox = image[y-padding:y+h+padding, x-padding:x+w+padding]
        # 테두리 잔여물 제거
        checkbox = clean_frame_border(checkbox)
        output_path = os.path.join(output_dir, f"checkbox_{i}.jpg")
        cv2.imwrite(output_path, checkbox)
        print(f"체크란 저장됨: {output_path}")
    
    # 유효한 프레임 필터링
    valid_frames = []
    for x, y, w, h in all_frames:
        # 프레임 크기 범위 내의 것만 선택
        if (avg_width * 0.85 <= w <= avg_width * 1.15 and 
            avg_height * 0.85 <= h <= avg_height * 1.15):
            valid_frames.append((x, y, w, h))
    
    # y 좌표로 행 그룹화
    row_groups = {}
    y_tolerance = avg_height * 0.3  # 같은 행으로 인식할 y좌표 차이
    
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
            # 테두리 잔여물 제거
            frame = clean_frame_border(frame)
            repeat_suffix = {1: "03", 2: "02", 3: "01"}[col_idx]  # 역순으로 변경
            output_path = os.path.join(output_dir, f"frame_{selected_q2}_{row_idx}_{repeat_suffix}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path}")
        
        # 오른쪽 프레임 처리 (문제1)
        for col_idx, (x, y, w, h) in enumerate(sorted(right_frames, key=lambda f: f[0]), 1):
            if col_idx > 3: continue
            padding = 5
            frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
            # 테두리 잔여물 제거
            frame = clean_frame_border(frame)
            repeat_suffix = {1: "03", 2: "02", 3: "01"}[col_idx]  # 역순으로 변경
            output_path = os.path.join(output_dir, f"frame_{selected_q1}_{row_idx}_{repeat_suffix}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path}")
    
    print("\n프레임 추출 완료")

    # 템플릿 출력 함수 호출
    if template_path and template_name:
        create_template_output(image, template_path, output_dir, template_name) 
import cv2
import numpy as np
import os

def clean_frame_border(frame):
    """
    프레임 이미지의 테두리 잔여물을 제거합니다.
    """
    # 이미지 유효성 검사
    if frame is None or frame.size == 0:
        print("경고: 유효하지 않은 이미지가 입력되었습니다.")
        return None
    
    # 그레이스케일 변환
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
    except cv2.error as e:
        print(f"이미지 변환 중 오류 발생: {e}")
        return None
    
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

def create_template_output(image, template_path, output_dir, template_name, frame_data=None):
    """
    템플릿 이미지에 프레임을 배치하여 출력합니다.
    각 문제의 21개 이미지를 해당 문제의 시작 위치부터 순차적으로 배치합니다.
    템플릿 이미지는 5000x5000px 크기로 고정됩니다.
    프레임 크기: 130x130px, 간격: 20px
    
    Args:
        frame_data: 프레임 정보 딕셔너리 (선택사항)
                   형식: {question_num: {row_idx: {position: frame_path}}}
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
    
    # 입력 이미지 유효성 검사
    if image is None or image.size == 0:
        raise ValueError("입력 이미지가 비어있거나 유효하지 않습니다.")
    
    # 템플릿 크기를 5000x5000으로 고정
    template = cv2.resize(template, (5000, 5000))
    output = template.copy()
    
    # 템플릿 프레임 설정
    frame_size = 130
    rows = 33
    cols = 34
    
    # 시작 위치 계산 (중앙 정렬)
    start_x = (template.shape[1] - (cols * (frame_size))) // 2
    start_y = (template.shape[0] - (rows * (frame_size))) // 2
    
    # 프레임 데이터 처리
    if frame_data is not None:
        # 직접 전달받은 프레임 데이터 사용
        question_files = frame_data
        print("직접 전달받은 프레임 데이터를 사용합니다.")
    else:
        # 기존 방식: 파일 목록에서 추출
        extracted_files = []
        for file in os.listdir(output_dir):
            if file.startswith('frame_question') and file.endswith('.jpg') and '_origin' not in file:
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
        
        # 각 question 번호별로 파일 그룹화 (기존 방식)
        question_files = {}
        for file_info in extracted_files:
            question = file_info[1]
            if question not in question_files:
                question_files[question] = []
            question_files[question].append(file_info)
    
    # 문제 번호별 시작 위치 계산
    question_start_positions = {
        "01": (1, 0),    # 1번 문제: 1번째 줄, 1번째 칸부터
        "02": (2, 8),   # 2번 문제: 2번째 줄, 9번째 칸부터
        "03": (3, 16),   # 3번 문제: 3번째 줄, 17번째 칸부터
        "04": (4, 24),   # 4번 문제: 4번째 줄, 25번째 칸부터
    }
    
    # 각 question 번호별로 프레임 배치 (01부터 04까지 순서대로)
    for question_num in sorted(question_files.keys()):
        if question_num not in question_start_positions:
            print(f"경고: 알 수 없는 문제 번호 {question_num}는 건너뜁니다.")
            continue
            
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
            template_frames.append((x, y, frame_size, frame_size, frame_count))
            frame_count += 1
            
            # 다음 위치 계산
            current_col += 2
            if current_col >= cols:  # 열이 끝나면 다음 행으로
                current_col = 0
                current_row += 1
        
        # 프레임 배치 - 위치 기반 방식 (누락 고려)
        if frame_data is not None:
            # 직접 전달받은 프레임 데이터 사용
            question_data = question_files[question_num]
            
            # 각 행별로 정확한 템플릿 위치 계산
            template_position = 0
            
            # 행별로 처리 (1~7행, 각 행당 3개 위치)
            for row_idx in range(1, 8):
                # 각 위치별로 처리 (왼쪽 3,2,1 또는 오른쪽 6,5,4)
                for pos_order in [3, 2, 1]:  # 역순으로 처리
                    if template_position >= len(template_frames):
                        break
                    
                    # 해당 행과 위치에 프레임이 있는지 확인
                    frame_path = None
                    actual_position = None
                    
                    if row_idx in question_data:
                        row_data = question_data[row_idx]
                        
                        # 왼쪽 프레임 (1,2,3) 또는 오른쪽 프레임 (4,5,6) 확인
                        for check_pos in [pos_order, pos_order + 3]:  # 1,4 / 2,5 / 3,6
                            if check_pos in row_data:
                                frame_path = row_data[check_pos]
                                actual_position = check_pos
                                break
                    
                    # 프레임이 있는 경우 배치
                    if frame_path and template_position < len(template_frames):
                        frame = cv2.imread(frame_path)
                        if frame is not None and frame.size > 0:
                            # 템플릿의 해당 위치
                            tx, ty, tw, th, template_idx = template_frames[template_position]
                            
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
                                # 출력 이미지 범위 체크
                                if (y_offset >= 0 and y_offset + new_height <= output.shape[0] and
                                    x_offset >= 0 and x_offset + new_width <= output.shape[1]):
                                    output[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
                                    print(f"프레임 배치: {os.path.basename(frame_path)} -> 템플릿 위치 {template_position} (행:{row_idx}, 위치:{actual_position})")
                                else:
                                    print(f"프레임 배치 실패: 범위를 벗어남 - {os.path.basename(frame_path)}")
                            except ValueError as e:
                                print(f"프레임 배치 오류: {e}")
                        else:
                            print(f"프레임 로드 실패: {frame_path}")
                    else:
                        # 프레임이 없는 경우 빈 공간으로 남김
                        print(f"빈 공간: 템플릿 위치 {template_position} (행:{row_idx}, 순서:{pos_order}) - 프레임 없음")
                    
                    template_position += 1
                
                if template_position >= len(template_frames):
                    break
        else:
            # 기존 방식: 파일 목록 기반
            files = question_files[question_num]
            for i, file_info in enumerate(files):
                if i >= len(template_frames):
                    break
                    
                file_name = file_info[0]
                frame_path = os.path.join(output_dir, file_name)
                
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"프레임을 로드할 수 없습니다: {frame_path}")
                    continue
                
                # 프레임 유효성 검사
                if frame.size == 0:
                    print(f"프레임이 비어있습니다: {frame_path}")
                    continue
                    
                # 템플릿의 현재 프레임 위치
                tx, ty, tw, th, template_idx = template_frames[i]
                
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
                    # 출력 이미지 범위 체크
                    if (y_offset >= 0 and y_offset + new_height <= output.shape[0] and
                        x_offset >= 0 and x_offset + new_width <= output.shape[1]):
                        output[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
                        print(f"프레임 배치: {file_name} -> 템플릿 위치 {template_idx}")
                    else:
                        print(f"프레임 배치 실패: 범위를 벗어남 - {file_name}")
                except ValueError as e:
                    print(f"프레임 배치 오류: {e}")
                    continue
    
    # 결과 이미지 저장 (output 디렉토리에 저장)
    output_path = os.path.join("output", f"{template_name}.jpg")
    
    # output 디렉토리가 없으면 생성
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 결과 이미지 저장 전 유효성 검사
    if output is not None and output.size > 0:
        try:
            cv2.imwrite(output_path, output)
            print(f"템플릿 출력 이미지 저장됨: {output_path}")
        except cv2.error as e:
            print(f"템플릿 출력 이미지 저장 중 오류 발생: {e}")
    else:
        print("경고: 출력 이미지가 유효하지 않습니다.")
    
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
    print("\n=== 이미지 처리 시작 ===")
    print(f"입력 이미지 shape: {image.shape if image is not None else 'None'}")
    print(f"입력 이미지 size: {image.size if image is not None else 0}")
    
    # 입력 이미지 유효성 검사
    if image is None or image.size == 0:
        print("경고: 입력 이미지가 비어있거나 유효하지 않습니다.")
        return None
    
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
    
    # 이미지 전처리
    try:
        if len(image.shape) == 3:
            print("3차원 이미지를 그레이스케일로 변환합니다.")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print("2차원 이미지를 복사합니다.")
            gray = image.copy()
        
        print(f"그레이스케일 이미지 shape: {gray.shape}")
        print(f"그레이스케일 이미지 size: {gray.size}")
        
    except cv2.error as e:
        print(f"이미지 변환 중 오류 발생: {e}")
        print(f"이미지 shape: {image.shape if image is not None else 'None'}")
        return None
    
    # 적응형 이진화 적용
    try:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        print(f"이진화 이미지 shape: {binary.shape}")
        print(f"이진화 이미지 size: {binary.size}")
    except cv2.error as e:
        print(f"이진화 중 오류 발생: {e}")
        return None
    
    # 수직선과 수평선 감지를 위한 구조화 요소
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # 수직선 감지
    try:
        vertical_lines = cv2.erode(binary, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)
        print(f"수직선 이미지 shape: {vertical_lines.shape}")
    except cv2.error as e:
        print(f"수직선 감지 중 오류 발생: {e}")
        return None
    
    # 수평선 감지
    try:
        horizontal_lines = cv2.erode(binary, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)
        print(f"수평선 이미지 shape: {horizontal_lines.shape}")
    except cv2.error as e:
        print(f"수평선 감지 중 오류 발생: {e}")
        return None
    
    # 격자 구조 결합
    try:
        grid = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # 모폴로지 연산으로 격자 선 강화
        kernel = np.ones((3,3), np.uint8)
        grid = cv2.dilate(grid, kernel, iterations=1)
        grid = cv2.erode(grid, kernel, iterations=1)
        print(f"격자 이미지 shape: {grid.shape}")
    except cv2.error as e:
        print(f"격자 구조 결합 중 오류 발생: {e}")
        return None
    
    # 디버깅을 위해 전처리된 이미지들 저장
    try:
        if binary is not None and binary.size > 0:
            cv2.imwrite(os.path.join(output_dir, 'debug_binary.jpg'), binary)
            print("이진화 이미지 저장 완료")
        if vertical_lines is not None and vertical_lines.size > 0:
            cv2.imwrite(os.path.join(output_dir, 'debug_vertical.jpg'), vertical_lines)
            print("수직선 이미지 저장 완료")
        if horizontal_lines is not None and horizontal_lines.size > 0:
            cv2.imwrite(os.path.join(output_dir, 'debug_horizontal.jpg'), horizontal_lines)
            print("수평선 이미지 저장 완료")
        if grid is not None and grid.size > 0:
            cv2.imwrite(os.path.join(output_dir, 'debug_grid.jpg'), grid)
            print("격자 이미지 저장 완료")
    except cv2.error as e:
        print(f"디버그 이미지 저장 중 오류 발생: {e}")
    
    # 원본 이미지 로드 (debug_resized.jpg)
    original_image = cv2.imread(os.path.join(output_dir, 'debug_resized.jpg'))
    if original_image is None:
        print("경고: debug_resized.jpg를 찾을 수 없습니다. 원본 프레임 추출을 건너뜁니다.")
    
    # 윤곽선 찾기
    contours, hierarchy = cv2.findContours(
        grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 답안 프레임 필터링
    all_frames = []
    min_area = image.shape[0] * image.shape[1] * 0.001  # 최소 영역 크기
    max_area = image.shape[0] * image.shape[1] * 0.02   # 최대 영역 크기
    
    # 프레임 중복 제거를 위한 최소 거리 설정
    min_distance = min(image.shape[0], image.shape[1]) * 0.05  # 이미지 크기의 5%
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w/h if h != 0 else 0
            
            # 정사각형에 가까운 프레임만 선택
            if 0.8 < aspect_ratio < 1.2:
                # 기존 프레임과의 거리 확인
                is_duplicate = False
                for existing_x, existing_y, _, _ in all_frames:
                    distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if distance < min_distance:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_frames.append((x, y, w, h))
    
    print(f"\n감지된 모든 프레임 수: {len(all_frames)}")
    
    # 디버깅을 위해 감지된 프레임 시각화
    debug_frame = image.copy()
    for i, (x, y, w, h) in enumerate(all_frames):
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 프레임 번호 표시
        cv2.putText(debug_frame, str(i+1), (x+5, y+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 디버그 이미지 저장 전 유효성 검사
    if debug_frame is not None and debug_frame.size > 0:
        try:
            cv2.imwrite(os.path.join(output_dir, 'debug_frames.jpg'), debug_frame)
            print(f"전체 프레임 감지 이미지 저장됨: debug_frames.jpg (총 {len(all_frames)}개)")
        except cv2.error as e:
            print(f"디버그 프레임 이미지 저장 중 오류 발생: {e}")
    
    # 평균 프레임 크기 계산 (이상치 제거)
    frame_sizes = [(w, h) for _, _, w, h in all_frames]
    if not frame_sizes:  # 프레임이 하나도 없는 경우
        print("경고: 감지된 프레임이 없습니다.")
        return None
        
    sizes_array = np.array(frame_sizes)
    
    # IQR 방식으로 이상치 제거
    q1_width = np.percentile(sizes_array[:, 0], 25)
    q3_width = np.percentile(sizes_array[:, 0], 75)
    iqr_width = q3_width - q1_width
    lower_bound_width = q1_width - 2.0 * iqr_width
    upper_bound_width = q3_width + 2.0 * iqr_width
    
    q1_height = np.percentile(sizes_array[:, 1], 25)
    q3_height = np.percentile(sizes_array[:, 1], 75)
    iqr_height = q3_height - q1_height
    lower_bound_height = q1_height - 2.0 * iqr_height
    upper_bound_height = q3_height + 2.0 * iqr_height
    
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
    
    # 이상치 제거 후 프레임 시각화
    debug_valid_frames = image.copy()
    for x, y, w, h in all_frames:
        if (lower_bound_width <= w <= upper_bound_width and 
            lower_bound_height <= h <= upper_bound_height):
            # 유효한 프레임은 녹색으로 표시
            cv2.rectangle(debug_valid_frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 프레임 크기 표시
            cv2.putText(debug_valid_frames, f"{w}x{h}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # 이상치로 제거된 프레임은 빨간색으로 표시
            cv2.rectangle(debug_valid_frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # 프레임 크기 표시
            cv2.putText(debug_valid_frames, f"{w}x{h}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 디버그 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_valid_frames.jpg'), debug_valid_frames)
    print("이상치 제거 후 프레임 디버그 이미지 저장됨: debug_valid_frames.jpg")
    
    # 체크란 크기 범위 설정 (답안 프레임의 50% 기준, ±10% 허용)
    checkbox_size = avg_width * 0.5
    checkbox_range = (int(checkbox_size * 0.9), int(checkbox_size * 1.1))
    
    # 체크란 감지 - 원본 이미지 사용
    if original_image is not None:
        checkbox_image = original_image
    else:
        checkbox_image = image
    
    height = checkbox_image.shape[0]
    top_area = checkbox_image[:height//4, :]  # 상단 25% 영역만 처리
    
    # 검은색 체크란 감지를 위한 이진화
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([100, 100, 100])  # 더 넓은 범위로 확장
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
        # 크기 조건 완화: 0.3~0.7배, 비율 0.7~1.3
        if (avg_width*0.3 <= w <= avg_width*0.7 and avg_height*0.3 <= h <= avg_height*0.7 and 0.7 < aspect_ratio < 1.3):
            padding_ratio = 0.2
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
            roi = top_area[y+pad_y:y+h-pad_y, x+pad_x:x+w-pad_x]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, v_thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(v_thresh == 255)
            total_pixels = (w-2*pad_x) * (h-2*pad_y)
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0
            print(f"체크란 후보 {i}: 좌표=({x},{y}), 크기={w}x{h}, 비율={aspect_ratio:.2f}, 흰색 비율={white_ratio:.3f}")
            checkboxes.append({
                'position': (x, y),
                'size': (w, h),
                'white_ratio': white_ratio,
                'is_checked': False,
                'number': None
            })
    # x좌표 기준으로 체크란 위치 분류
    if len(checkboxes) >= 4:
        # x좌표 기준으로 정렬
        checkboxes.sort(key=lambda box: box['position'][0])
        
        # 왼쪽 2개 (02, 04) - x좌표가 작은 순서
        left_boxes = sorted(checkboxes[:2], key=lambda box: box['position'][0])  # x좌표로 정렬
        # 02가 04보다 왼쪽에 있음
        left_boxes[0]['number'] = 'question02'  # 가장 왼쪽
        left_boxes[1]['number'] = 'question04'  # 두 번째 왼쪽
        
        # 오른쪽 2개 (01, 03) - x좌표가 큰 순서  
        right_boxes = sorted(checkboxes[-2:], key=lambda box: box['position'][0])  # x좌표로 정렬
        # 01이 03보다 왼쪽에 있음 (오른쪽 그룹 내에서)
        right_boxes[0]['number'] = 'question01'  # 오른쪽 그룹의 왼쪽
        right_boxes[1]['number'] = 'question03'  # 오른쪽 그룹의 오른쪽
        
        # 최종 체크박스 배열: [question02, question04, question01, question03]
        checkboxes = [left_boxes[0], left_boxes[1], right_boxes[0], right_boxes[1]]
        
        print(f"\n=== 체크란 위치 분류 결과 ===")
        for box in checkboxes:
            x, y = box['position']
            print(f"{box['number']}: x={x}, y={y}")
        print("=========================")
        
    else:
        print(f"[경고] 체크란 후보가 4개가 아님. 후보 전체: {len(checkboxes)}개")
        for i, box in enumerate(checkboxes):
            print(f"  후보{i}: 좌표={box['position']}, 크기={box['size']}, 흰색비율={box['white_ratio']:.3f}")
    # 체크 여부 판단
    if len(checkboxes) == 4:
        # 問一 vs 問三
        if checkboxes[2]['white_ratio'] < checkboxes[3]['white_ratio']:
            checkboxes[2]['is_checked'] = True
        else:
            checkboxes[3]['is_checked'] = True
        # 問二 vs 問四
        if checkboxes[0]['white_ratio'] < checkboxes[1]['white_ratio']:
            checkboxes[0]['is_checked'] = True
        else:
            checkboxes[1]['is_checked'] = True
    
    # 체크란 표시 및 결과 출력
    print("\n=== 체크란 감지 결과 ===")
    debug_checkbox_image = checkbox_image.copy()
    for box in checkboxes:
        x, y = box['position']
        w, h = box['size']
        color = (0, 255, 0) if box['is_checked'] else (0, 0, 255)
        cv2.rectangle(debug_checkbox_image, (x, y), (x+w, y+h), color, 2)
        
        status = "체크됨 ✓" if box['is_checked'] else "체크되지 않음 ✗"
        print(f"{box['number']}: {status} (흰색 비율: {box['white_ratio']:.3f})")
    
    print("\n=== 문제별 선택된 답안 ===")
    # 문제1(오른쪽, 問一/問三)은 checkboxes[2:4], 문제2(왼쪽, 問二/問四)는 checkboxes[0:2]
    selected_q1 = next((box['number'] for box in checkboxes[2:4] if box['is_checked']), 'question01')
    selected_q2 = next((box['number'] for box in checkboxes[0:2] if box['is_checked']), 'question04')
    print(f"문제1: {selected_q1}")
    print(f"문제2: {selected_q2}")
    print("=====================")
    
    # 디버깅을 위해 처리된 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'debug_checkboxes.jpg'), debug_checkbox_image)
    
    # 체크란 이미지 저장
    for i, box in enumerate(checkboxes, 1):
        x, y = box['position']
        w, h = box['size']
        padding = 2
        
        try:
            # 체크란 영역이 이미지 범위를 벗어나지 않는지 확인
            if (y-padding >= 0 and y+h+padding <= checkbox_image.shape[0] and 
                x-padding >= 0 and x+w+padding <= checkbox_image.shape[1]):
                
                # 체크란 영역 추출 (원본 이미지에서)
                checkbox = checkbox_image[y-padding:y+h+padding, x-padding:x+w+padding]
                
                # 추출된 영역이 유효한지 확인
                if checkbox is not None and checkbox.size > 0:
                    # 테두리 잔여물 제거
                    cleaned_checkbox = clean_frame_border(checkbox)
                    
                    if cleaned_checkbox is not None and cleaned_checkbox.size > 0:
                        output_path = os.path.join(output_dir, f"checkbox_{i}.jpg")
                        cv2.imwrite(output_path, cleaned_checkbox)
                        print(f"체크란 저장됨: {output_path}")
                    else:
                        print(f"경고: 체크란 {i} 정리 후 이미지가 유효하지 않습니다.")
                else:
                    print(f"경고: 체크란 {i} 영역이 유효하지 않습니다.")
            else:
                print(f"경고: 체크란 {i} 위치가 이미지 범위를 벗어났습니다.")
        except Exception as e:
            print(f"체크란 {i} 저장 중 오류 발생: {e}")
            continue
    
    # 유효한 프레임 필터링 (범위를 더 넓게 설정)
    valid_frames = []
    for x, y, w, h in all_frames:
        # 프레임 크기 범위를 더 넓게 설정 (0.7 ~ 1.3배)
        if (avg_width * 0.7 <= w <= avg_width * 1.3 and 
            avg_height * 0.7 <= h <= avg_height * 1.3):
            valid_frames.append((x, y, w, h))
    
    print(f"\n=== 프레임 필터링 결과 ===")
    print(f"전체 감지된 프레임 수: {len(all_frames)}")
    print(f"유효한 프레임 수: {len(valid_frames)}")
    print(f"평균 프레임 크기: {avg_width:.0f} x {avg_height:.0f}")
    print("=========================\n")
    
    # y 좌표로 행 그룹화 (허용 오차를 더 크게 설정)
    row_groups = {}
    y_tolerance = avg_height * 0.5  # 같은 행으로 인식할 y좌표 차이를 더 크게
    
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
    
    # 행 그룹화 결과 출력
    print(f"=== 행 그룹화 결과 (y_tolerance: {y_tolerance:.0f}) ===")
    for i, row_y in enumerate(sorted(row_groups.keys()), 1):
        frames_count = len(row_groups[row_y])
        print(f"행 {i} (y={row_y:.0f}): {frames_count}개 프레임")
        for j, (x, y, w, h) in enumerate(row_groups[row_y]):
            print(f"  프레임 {j+1}: x={x}, y={y}, 크기={w}x{h}")
    print("=" * 50)
    
    # 이미지 중간점 계산
    width = image.shape[1]
    mid_x = width // 2
    
    # 각 행 정렬 및 처리
    sorted_rows = []
    for row_y in sorted(row_groups.keys()):
        row_frames = row_groups[row_y]
        # x 좌표로 정렬하되, 왼쪽에서 오른쪽으로
        row_frames.sort(key=lambda f: f[0])
        sorted_rows.append(row_frames)
    
    # 7행만 선택하고 각 행의 프레임을 왼쪽/오른쪽으로 분류
    sorted_rows = sorted_rows[:7]
    
    # 원본 이미지 로드 (debug_resized.jpg)
    original_image = cv2.imread(os.path.join(output_dir, 'debug_resized.jpg'))
    if original_image is None:
        print("경고: debug_resized.jpg를 찾을 수 없습니다. 원본 프레임 추출을 건너뜁니다.")
    
    def assign_frame_position_by_coordinate(frames, expected_positions=6):
        """X좌표를 기준으로 실제 위치를 계산하여 매핑"""
        if not frames:
            return {}
        
        # 전체 이미지 너비를 기준으로 예상 위치 계산
        image_width = image.shape[1]
        
        # 6개 프레임의 예상 위치 (이미지 너비를 6등분)
        expected_x_positions = []
        for i in range(expected_positions):
            expected_x = (image_width / (expected_positions + 1)) * (i + 1)
            expected_x_positions.append(expected_x)
        
        print(f"  예상 X 위치들: {[int(x) for x in expected_x_positions]}")
        
        position_map = {}
        for frame in frames:
            x, y, w, h = frame
            frame_center_x = x + w // 2  # 프레임 중심점 사용
            
            # 가장 가까운 예상 위치 찾기
            min_distance = float('inf')
            best_position = 1
            
            for pos, expected_x in enumerate(expected_x_positions, 1):
                distance = abs(frame_center_x - expected_x)
                if distance < min_distance:
                    min_distance = distance
                    best_position = pos
            
            # 거리 임계값 확인 (너무 멀리 떨어진 경우 제외)
            max_distance = image_width * 0.15  # 이미지 너비의 15%
            if min_distance <= max_distance:
                # 이미 해당 위치에 프레임이 있는 경우, 더 가까운 것을 선택
                if best_position in position_map:
                    existing_frame = position_map[best_position]
                    existing_center_x = existing_frame[0] + existing_frame[2] // 2
                    existing_distance = abs(existing_center_x - expected_x_positions[best_position - 1])
                    
                    if min_distance < existing_distance:
                        position_map[best_position] = frame
                        print(f"  프레임 x={x} -> 위치 {best_position} (기존 프레임 교체, 거리: {min_distance:.1f})")
                    else:
                        print(f"  프레임 x={x} -> 위치 {best_position} 건너뛰기 (기존 프레임이 더 가까움)")
                else:
                    position_map[best_position] = frame
                    print(f"  프레임 x={x} -> 위치 {best_position} (거리: {min_distance:.1f})")
            else:
                print(f"  프레임 x={x} -> 위치 매핑 실패 (거리 {min_distance:.1f} > 임계값 {max_distance:.1f})")
        
        return position_map

    for row_idx, row in enumerate(sorted_rows, 1):
        # 프레임을 x 좌표 기준으로 정렬
        sorted_frames = sorted(row, key=lambda f: f[0])
        
        print(f"\n--- 행 {row_idx} 처리 ---")
        print(f"이 행의 총 프레임 수: {len(sorted_frames)}")
        
        # 물리적 위치 기반으로 프레임 매핑
        if len(sorted_frames) > 0:
            # 전체 프레임의 위치 매핑 (1~6)
            position_map = assign_frame_position_by_coordinate(sorted_frames, 6)
            
            # 왼쪽 프레임 (1, 2, 3번 위치)
            left_positions = {}
            for pos in [1, 2, 3]:
                if pos in position_map:
                    left_positions[pos] = position_map[pos]
            
            # 오른쪽 프레임 (4, 5, 6번 위치)
            right_positions = {}
            for pos in [4, 5, 6]:
                if pos in position_map:
                    right_positions[pos] = position_map[pos]
            
            print(f"왼쪽 프레임 위치: {list(left_positions.keys())}")
            print(f"오른쪽 프레임 위치: {list(right_positions.keys())}")
            
            # 각 프레임의 실제 위치와 x 좌표 출력
            for pos, (x, y, w, h) in left_positions.items():
                print(f"  왼쪽 위치{pos}: x={x}, y={y}")
            for pos, (x, y, w, h) in right_positions.items():
                print(f"  오른쪽 위치{pos}: x={x}, y={y}")
        else:
            left_positions = {}
            right_positions = {}
            print("이 행에는 프레임이 없습니다.")
        
        # 디버그 이미지 생성
        debug_frame_positions = image.copy()
        
        # 왼쪽 프레임 처리 (문제2) - 위치 기반 처리
        for position, (x, y, w, h) in left_positions.items():
            padding = 5
            
            # 프레임 위치 표시 (녹색)
            cv2.rectangle(debug_frame_positions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 프레임 번호 표시 (실제 위치)
            cv2.putText(debug_frame_positions, f"L{row_idx}-{position}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 전처리된 이미지에서 프레임 추출
            frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
            frame = clean_frame_border(frame)
            frame_height, frame_width = frame.shape[:2]
            
            # 위치에 따른 suffix 매핑 (1->03, 2->02, 3->01)
            suffix_map = {1: "03", 2: "02", 3: "01"}
            repeat_suffix = suffix_map[position]
            base_name = f"frame_{selected_q2}_{row_idx}_{repeat_suffix}"
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path} (실제 위치: {position})")
            
            if original_image is not None:
                original_frame = original_image[y+padding:y+h-padding, x+padding:x+w-padding]
                if original_frame.shape[:2] != (frame_height, frame_width):
                    original_frame = cv2.resize(original_frame, (frame_width, frame_height))
                original_path = os.path.join(output_dir, f"{base_name}_origin.jpg")
                cv2.imwrite(original_path, original_frame)
                print(f"원본 프레임 저장됨: {original_path}")
        
        # 오른쪽 프레임 처리 (문제1) - 위치 기반 처리
        for position, (x, y, w, h) in right_positions.items():
            padding = 5
            
            # 프레임 위치 표시 (녹색)
            cv2.rectangle(debug_frame_positions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 프레임 번호 표시 (실제 위치)
            cv2.putText(debug_frame_positions, f"R{row_idx}-{position}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 전처리된 이미지에서 프레임 추출
            frame = image[y+padding:y+h-padding, x+padding:x+w-padding]
            frame = clean_frame_border(frame)
            frame_height, frame_width = frame.shape[:2]
            
            # 위치에 따른 suffix 매핑 (4->03, 5->02, 6->01)
            suffix_map = {4: "03", 5: "02", 6: "01"}
            repeat_suffix = suffix_map[position]
            base_name = f"frame_{selected_q1}_{row_idx}_{repeat_suffix}"
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"저장됨: {output_path} (실제 위치: {position})")
            
            if original_image is not None:
                original_frame = original_image[y+padding:y+h-padding, x+padding:x+w-padding]
                if original_frame.shape[:2] != (frame_height, frame_width):
                    original_frame = cv2.resize(original_frame, (frame_width, frame_height))
                original_path = os.path.join(output_dir, f"{base_name}_origin.jpg")
                cv2.imwrite(original_path, original_frame)
                print(f"원본 프레임 저장됨: {original_path}")
        
        # 누락된 프레임 위치 보고
        all_expected_positions = set(range(1, 7))
        detected_positions = set(left_positions.keys()) | set(right_positions.keys())
        missing_positions = all_expected_positions - detected_positions
        if missing_positions:
            print(f"⚠️  누락된 프레임 위치: {sorted(missing_positions)}")
        
        # 통합 디버그 이미지를 위한 프레임 정보 수집
        if 'all_frame_positions' not in locals():
            all_frame_positions = []
        
        # 왼쪽 프레임 정보 추가
        for position, (x, y, w, h) in left_positions.items():
            all_frame_positions.append((x, y, w, h, f"L{row_idx}-{position}"))
        
        # 오른쪽 프레임 정보 추가
        for position, (x, y, w, h) in right_positions.items():
            all_frame_positions.append((x, y, w, h, f"R{row_idx}-{position}"))
        
        # 개별 행 디버그 이미지도 저장 (기존 방식 유지)
        debug_path = os.path.join(output_dir, f'debug_frame_positions_row_{row_idx}.jpg')
        cv2.imwrite(debug_path, debug_frame_positions)
        print(f"프레임 위치 디버그 이미지 저장됨: {debug_path}")
    
    # 원본 이미지 한 장에 모든 프레임 표시
    if 'all_frame_positions' in locals() and all_frame_positions:
        print(f"\n=== 통합 프레임 위치 표시 ===")
        
        # 원본 이미지 복사
        combined_debug = image.copy()
        
        # 이미지가 그레이스케일인 경우 컬러로 변환
        if len(combined_debug.shape) == 2:
            combined_debug = cv2.cvtColor(combined_debug, cv2.COLOR_GRAY2BGR)
        
        # 모든 프레임 위치 표시
        for x, y, w, h, label in all_frame_positions:
            # 프레임 사각형 그리기 (녹색)
            cv2.rectangle(combined_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 라벨 텍스트 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # 텍스트 배경 (검은색)
            cv2.rectangle(combined_debug, (x, y-25), (x+text_size[0]+10, y-5), (0, 0, 0), -1)
            # 텍스트 (흰색)
            cv2.putText(combined_debug, label, (x+5, y-10), font, font_scale, (255, 255, 255), font_thickness)
        
        # 통합 프레임 위치 이미지 저장
        combined_path = os.path.join(output_dir, 'debug_all_frame_positions.jpg')
        try:
            cv2.imwrite(combined_path, combined_debug)
            print(f"통합 프레임 위치 이미지 저장됨: {combined_path}")
            print(f"총 {len(all_frame_positions)}개 프레임 표시")
        except cv2.error as e:
            print(f"통합 프레임 위치 이미지 저장 중 오류: {e}")
        
        print("=" * 40)
    
    print("\n프레임 추출 완료")

    # 템플릿 출력 함수 호출
    if template_path and template_name:
        # 프레임 데이터 수집
        frame_data = {}
        
        # 선택된 문제별로 프레임 데이터 구성
        print(f"선택된 문제: Q1={selected_q1}, Q2={selected_q2}")
        
        # 각 문제별로 별도 처리
        q1_num = selected_q1.replace('question', '')
        q2_num = selected_q2.replace('question', '')
        
        # 문제1 데이터 (오른쪽 프레임)
        if q1_num in ['01', '02', '03', '04']:
            frame_data[q1_num] = {}
            for row_idx in range(1, 8):  # 1~7행
                row_frame_data = {}
                
                # 오른쪽 프레임 (4, 5, 6번 위치)
                for position in [4, 5, 6]:
                    suffix_map = {4: "03", 5: "02", 6: "01"}
                    frame_name = f"frame_{selected_q1}_{row_idx}_{suffix_map[position]}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    if os.path.exists(frame_path):
                        row_frame_data[position] = frame_path
                        print(f"Q1 프레임 추가: 행{row_idx}, 위치{position} -> {frame_name}")
                
                # 행에 프레임이 있는 경우만 추가
                if row_frame_data:
                    frame_data[q1_num][row_idx] = row_frame_data
        
        # 문제2 데이터 (왼쪽 프레임) - 문제1과 다른 경우만
        if q2_num in ['01', '02', '03', '04'] and q2_num != q1_num:
            frame_data[q2_num] = {}
            for row_idx in range(1, 8):  # 1~7행
                row_frame_data = {}
                
                # 왼쪽 프레임 (1, 2, 3번 위치)
                for position in [1, 2, 3]:
                    suffix_map = {1: "03", 2: "02", 3: "01"}
                    frame_name = f"frame_{selected_q2}_{row_idx}_{suffix_map[position]}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    if os.path.exists(frame_path):
                        row_frame_data[position] = frame_path
                        print(f"Q2 프레임 추가: 행{row_idx}, 위치{position} -> {frame_name}")
                
                # 행에 프레임이 있는 경우만 추가
                if row_frame_data:
                    frame_data[q2_num][row_idx] = row_frame_data
        elif q2_num == q1_num:
            # 같은 문제 번호인 경우 왼쪽 프레임도 추가
            for row_idx in range(1, 8):  # 1~7행
                if row_idx not in frame_data[q1_num]:
                    frame_data[q1_num][row_idx] = {}
                
                # 왼쪽 프레임 (1, 2, 3번 위치) 추가
                for position in [1, 2, 3]:
                    suffix_map = {1: "03", 2: "02", 3: "01"}
                    frame_name = f"frame_{selected_q2}_{row_idx}_{suffix_map[position]}.jpg"
                    frame_path = os.path.join(output_dir, frame_name)
                    if os.path.exists(frame_path):
                        frame_data[q1_num][row_idx][position] = frame_path
                        print(f"Q1+Q2 프레임 추가: 행{row_idx}, 위치{position} -> {frame_name}")
        
        print(f"\n=== 템플릿 출력용 프레임 데이터 ===")
        for question_num, question_data in frame_data.items():
            print(f"문제 {question_num}: {len(question_data)}개 행")
            for row_idx, row_data in question_data.items():
                print(f"  행 {row_idx}: 위치 {list(row_data.keys())}")
        print("=" * 40)
        
        # 개선된 템플릿 출력 함수 호출
        create_template_output(image, template_path, output_dir, template_name, frame_data) 
import os
import cv2
import numpy as np
from writing_detector import clean_frame_border


def _detect_marked_question_repeat(image, output_dir):
    """
    Detect top checkboxes:
    - Upper row 5: question 1/2/3/4/5
    - Lower row 3: repeat 1/2/3
    Returns (question_idx, repeat_idx) with 1-based indexes.
    """
    h, w = image.shape[:2]
    top_h = int(h * 0.26)
    top_area = image[:top_h, :]

    gray = cv2.cvtColor(top_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < int(w * 0.018) or bw > int(w * 0.11):
            continue
        if bh < int(h * 0.012) or bh > int(h * 0.09):
            continue
        ratio = bw / bh if bh else 0
        if ratio < 0.7 or ratio > 1.3:
            continue
        if y > int(top_h * 0.85):
            continue
        candidates.append((x, y, bw, bh))

    expected_centers = [
        (0.19, 0.09), (0.35, 0.09), (0.50, 0.09), (0.66, 0.09), (0.82, 0.09),
        (0.19, 0.19), (0.35, 0.19), (0.50, 0.19),
    ]
    default_size = max(14, int(min(w, h) * 0.045))
    slots = []

    for ex, ey in expected_centers:
        cx = int(w * ex)
        cy = int(h * ey)
        best = None
        best_dist = 1e9
        for rect in candidates:
            x, y, bw, bh = rect
            rcx = x + bw // 2
            rcy = y + bh // 2
            dist = ((rcx - cx) ** 2 + (rcy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = rect
        if best is not None and best_dist < w * 0.06:
            slots.append(best)
        else:
            x0 = max(0, cx - default_size // 2)
            y0 = max(0, cy - default_size // 2)
            slots.append((x0, y0, default_size, default_size))

    question_group = slots[:5]
    repeat_group = slots[5:]

    def dark_ratio(rect):
        x, y, bw, bh = rect
        px = max(1, int(bw * 0.2))
        py = max(1, int(bh * 0.2))
        roi = gray[y + py:y + bh - py, x + px:x + bw - px]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi < 170))

    question_scores = [dark_ratio(r) for r in question_group]
    repeat_scores = [dark_ratio(r) for r in repeat_group]

    question_idx = int(np.argmax(question_scores)) + 1
    repeat_idx = int(np.argmax(repeat_scores)) + 1

    debug = image.copy()
    for (x, y, bw, bh) in candidates:
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (80, 80, 80), 1)
    for i, (x, y, bw, bh) in enumerate(question_group, start=1):
        color = (0, 255, 0) if i == question_idx else (0, 0, 255)
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(debug, f"Q{i}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, (x, y, bw, bh) in enumerate(repeat_group, start=1):
        color = (255, 255, 0) if i == repeat_idx else (255, 0, 0)
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(debug, f"R{i}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    debug_path = os.path.join(output_dir, "debug_four_character_checkboxes.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"체크박스 디버그 저장됨: {debug_path}")
    print(f"선택 결과: question{question_idx:02d}, repeat={repeat_idx}")
    return question_idx, repeat_idx


def _fallback_grid_boxes(image):
    h, w = image.shape[:2]
    x_margin = int(w * 0.045)
    # New sheet has a taller top header (question/repeat rows).
    y_top = int(h * 0.24)
    y_bottom = int(h * 0.88)
    body_h = y_bottom - y_top
    row_h = body_h // 2
    col_w = (w - x_margin * 2) // 2

    boxes = []
    for row in range(2):
        for col in range(2):
            x = x_margin + col * col_w
            y = y_top + row * row_h
            bw = col_w - int(w * 0.01)
            bh = row_h - int(h * 0.03)
            boxes.append((x, y, bw, bh))
    return boxes


def _detect_four_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    area_min = h * w * 0.06
    area_max = h * w * 0.30
    candidates = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        area = bw * bh
        if area < area_min or area > area_max:
            continue
        ratio = bw / bh if bh else 0
        if ratio < 0.7 or ratio > 1.3:
            continue
        # New layout places lower boxes closer to the footer area.
        if y < int(h * 0.15) or y + bh > int(h * 0.97):
            continue
        candidates.append((x, y, bw, bh))

    candidates = sorted(candidates, key=lambda r: r[2] * r[3], reverse=True)
    selected = []
    for rect in candidates:
        x, y, bw, bh = rect
        cx, cy = x + bw // 2, y + bh // 2
        too_close = False
        for sx, sy, sw, sh in selected:
            scx, scy = sx + sw // 2, sy + sh // 2
            if abs(cx - scx) < bw * 0.3 and abs(cy - scy) < bh * 0.3:
                too_close = True
                break
        if not too_close:
            selected.append(rect)
        if len(selected) == 4:
            break

    if len(selected) < 4:
        return _fallback_grid_boxes(image)

    selected.sort(key=lambda r: (r[1], r[0]))
    top_row = sorted(selected[:2], key=lambda r: r[0])       # [left-top, right-top]
    bottom_row = sorted(selected[2:], key=lambda r: r[0])    # [left-bottom, right-bottom]

    # Four-character numbering order:
    # 1: right-top, 2: right-bottom, 3: left-top, 4: left-bottom
    left_top, right_top = top_row
    left_bottom, right_bottom = bottom_row
    return [right_top, right_bottom, left_top, left_bottom]


def _create_template_output(output_dir, template_path, template_name):
    if template_path is None:
        return

    template = cv2.imread(template_path)
    if template is None:
        output_template_path = os.path.join("output", os.path.basename(template_path))
        template = cv2.imread(output_template_path)
        if template is None:
            if os.path.basename(template_path) != "template.png":
                template = cv2.imread("template.png")
            if template is None:
                print(
                    "경고: 템플릿을 읽을 수 없어 템플릿 출력 생성을 건너뜁니다: "
                    f"{template_path}"
                )
                return

    template = cv2.resize(template, (5000, 5000))
    output = template.copy()

    frame_size = 130
    rows = 33
    cols = 34
    start_x = (template.shape[1] - (cols * frame_size)) // 2
    start_y = (template.shape[0] - (rows * frame_size)) // 2

    extracted_files = []
    for file in os.listdir(output_dir):
        if not (file.startswith("frame_question") and file.endswith("_origin.jpg")):
            continue
        parts = file.replace("_origin.jpg", "").split("_")
        if len(parts) < 4:
            continue
        question = parts[1].replace("question", "")
        if len(question) == 1:
            question = f"0{question}"
        try:
            repeat = int(parts[2])
            box = int(parts[3])
        except ValueError:
            continue
        extracted_files.append((file, question, repeat, box))

    extracted_files.sort(key=lambda x: (x[1], x[2], x[3]))
    if not extracted_files:
        print("경고: 템플릿 출력용 프레임 파일이 없습니다.")
        return

    question_files = {}
    for file_info in extracted_files:
        question = file_info[1]
        if question not in question_files:
            question_files[question] = []
        question_files[question].append(file_info)

    question_start_positions = {
        "01": (1, 0),
        "02": (2, 8),
        "03": (3, 16),
        "04": (4, 24),
        "05": (5, 32),
    }

    for question_num in sorted(question_files.keys()):
        if question_num not in question_start_positions:
            print(f"경고: 알 수 없는 문제 번호 {question_num}는 건너뜁니다.")
            continue

        q_start_row, q_start_col = question_start_positions[question_num]
        template_frames = []
        frame_count = 0
        current_row = q_start_row
        current_col = q_start_col
        while frame_count < 21 and current_row < rows:
            x = start_x + current_col * frame_size
            y = start_y + current_row * frame_size
            template_frames.append((x, y, frame_size, frame_size))
            frame_count += 1
            current_col += 2
            if current_col >= cols:
                current_col = 0
                current_row += 1

        question_entries = question_files[question_num]
        max_box = max(entry[3] for entry in question_entries) if question_entries else 1
        max_box = max(1, max_box)

        for file_name, _, repeat, box in question_entries:
            template_idx = (repeat - 1) * max_box + (box - 1)
            if template_idx < 0 or template_idx >= len(template_frames):
                print(f"경고: 템플릿 범위를 벗어난 프레임 건너뜀: {file_name}")
                continue

            frame_path = os.path.join(output_dir, file_name)
            frame = cv2.imread(frame_path)
            if frame is None or frame.size == 0:
                continue

            tx, ty, tw, th = template_frames[template_idx]
            target_w = tw - 4
            target_h = th - 4
            frame_h, frame_w = frame.shape[:2]
            frame_ratio = frame_w / frame_h
            target_ratio = target_w / target_h

            if frame_ratio > target_ratio:
                new_w = target_w
                new_h = max(1, int(new_w / frame_ratio))
            else:
                new_h = target_h
                new_w = max(1, int(new_h * frame_ratio))

            resized = cv2.resize(frame, (new_w, new_h))
            x_offset = tx + 2 + (tw - new_w) // 2
            y_offset = ty + 2 + (th - new_h) // 2
            output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            print(f"프레임 배치: {file_name} -> 문제 {question_num} 템플릿 순서 {template_idx + 1}")

    if template_name:
        filename = f"{template_name}.jpg"
    else:
        filename = "four_character_template_output.jpg"

    output_path = os.path.join("output", filename)
    cv2.imwrite(output_path, output)
    print(f"템플릿 출력 저장됨: {output_path}")


def four_character_detect_and_save_frames(image, output_dir, template_path=None, template_name=None):
    """
    Detect and save 4 large character boxes (2x2 layout) for four-character format.
    """
    if image is None or image.size == 0:
        raise ValueError("입력 이미지가 비어있거나 유효하지 않습니다.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    question_idx, repeat_idx = _detect_marked_question_repeat(image, output_dir)
    original_image = cv2.imread(os.path.join(output_dir, "debug_resized.jpg"))
    if original_image is None:
        print("경고: debug_resized.jpg를 찾을 수 없어 origin 저장에 전처리 이미지를 사용합니다.")
        original_image = image

    boxes = _detect_four_boxes(image)
    debug = image.copy()
    question_tag = f"question{question_idx:02d}"
    height, width = image.shape[:2]

    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        pad = max(4, int(min(bw, bh) * 0.01))
        x0, y0 = max(0, x + pad), max(0, y + pad)
        x1, y1 = min(width, x + bw - pad), min(height, y + bh - pad)
        raw_frame = image[y0:y1, x0:x1]
        origin_frame = original_image[y0:y1, x0:x1]
        if raw_frame.size == 0:
            continue
        if origin_frame is None or origin_frame.size == 0:
            origin_frame = raw_frame

        cleaned_frame = clean_frame_border(raw_frame)
        frame = cleaned_frame if cleaned_frame is not None and cleaned_frame.size > 0 else raw_frame
        if frame.size == 0:
            continue

        suffix = f"{i:02d}"
        frame_name = f"frame_{question_tag}_{repeat_idx}_{suffix}.jpg"
        origin_name = f"frame_{question_tag}_{repeat_idx}_{suffix}_origin.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        cv2.imwrite(os.path.join(output_dir, origin_name), origin_frame)

        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(debug, suffix, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    debug_path = os.path.join(output_dir, "debug_four_character_frames.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"4문자 프레임 디버그 저장됨: {debug_path}")

    _create_template_output(output_dir, template_path, template_name)

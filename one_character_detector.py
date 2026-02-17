import os
import cv2
import numpy as np


def _detect_marked_question_repeat(image, output_dir):
    """
    Detect 6 top checkboxes:
    - Left 3: question 1/2/3
    - Right 3: repeat 1/2/3
    Returns (question_idx, repeat_idx) with 1-based indexes.
    """
    h, w = image.shape[:2]
    top_h = int(h * 0.22)
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

    # Fixed top layout slots (left Q1-3, right R1-3).
    expected_centers = [
        (0.22, 0.09), (0.31, 0.09), (0.40, 0.09),
        (0.68, 0.09), (0.77, 0.09), (0.86, 0.09),
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

    left_group = slots[:3]
    right_group = slots[3:]

    def dark_ratio(rect):
        x, y, bw, bh = rect
        px = max(1, int(bw * 0.2))
        py = max(1, int(bh * 0.2))
        roi = gray[y + py:y + bh - py, x + px:x + bw - px]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi < 170))

    left_scores = [dark_ratio(r) for r in left_group]
    right_scores = [dark_ratio(r) for r in right_group]

    question_idx = int(np.argmax(left_scores)) + 1
    repeat_idx = int(np.argmax(right_scores)) + 1

    debug = image.copy()
    for (x, y, bw, bh) in candidates:
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (80, 80, 80), 1)
    for i, (x, y, bw, bh) in enumerate(left_group, start=1):
        color = (0, 255, 0) if i == question_idx else (0, 0, 255)
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(debug, f"Q{i}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, (x, y, bw, bh) in enumerate(right_group, start=1):
        color = (255, 255, 0) if i == repeat_idx else (255, 0, 0)
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(debug, f"R{i}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    debug_path = os.path.join(output_dir, "debug_one_character_checkboxes.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"체크박스 디버그 저장됨: {debug_path}")
    print(f"선택 결과: question{question_idx:02d}, repeat={repeat_idx}")
    return question_idx, repeat_idx


def _find_largest_inner_frame(image):
    """Find the largest inner rectangular frame used by one-character sheets."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    image_area = height * width

    best_rect = None
    best_area = 0

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        if area < image_area * 0.20:
            continue

        aspect_ratio = w / h if h else 0
        if aspect_ratio < 0.5 or aspect_ratio > 1.2:
            continue

        # One-character frame is below top checkboxes and above footer.
        if y < int(height * 0.10) or (y + h) > int(height * 0.92):
            continue

        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    return best_rect


def _create_template_output(frame_image, template_path, template_name):
    if template_path is None:
        return

    template = cv2.imread(template_path)
    if template is None:
        print(f"경고: 템플릿을 읽을 수 없어 템플릿 출력 생성을 건너뜁니다: {template_path}")
        return

    rect = _find_largest_inner_frame(template)
    if rect is None:
        th, tw = template.shape[:2]
        side = int(min(tw, th) * 0.7)
        x = (tw - side) // 2
        y = (th - side) // 2
        rect = (x, y, side, side)

    x, y, w, h = rect
    inner_pad = max(4, int(min(w, h) * 0.01))
    tx0, ty0 = x + inner_pad, y + inner_pad
    tw, th = max(1, w - inner_pad * 2), max(1, h - inner_pad * 2)

    resized = cv2.resize(frame_image, (tw, th))
    output = template.copy()
    output[ty0:ty0 + th, tx0:tx0 + tw] = resized

    if template_name:
        filename = f"{template_name}.png"
    else:
        filename = "one_character_template_output.png"

    output_path = os.path.join("output", filename)
    cv2.imwrite(output_path, output)
    print(f"템플릿 출력 저장됨: {output_path}")


def one_character_detect_and_save_frame(image, output_dir, template_path=None, template_name=None):
    """
    Detect and save the single large frame used by one-character answer sheets.
    """
    if image is None or image.size == 0:
        raise ValueError("입력 이미지가 비어있거나 유효하지 않습니다.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    question_idx, repeat_idx = _detect_marked_question_repeat(image, output_dir)

    rect = _find_largest_inner_frame(image)
    height, width = image.shape[:2]

    if rect is None:
        # Fallback: central crop that still captures the main writing area.
        crop_w = int(width * 0.78)
        crop_h = int(height * 0.62)
        x = (width - crop_w) // 2
        y = int(height * 0.20)
        rect = (x, y, crop_w, crop_h)
        print("경고: 프레임 검출 실패. 중앙 영역 fallback 크롭을 사용합니다.")

    x, y, w, h = rect
    inner_pad = max(4, int(min(w, h) * 0.01))
    x0, y0 = x + inner_pad, y + inner_pad
    x1, y1 = x + w - inner_pad, y + h - inner_pad

    frame = image[y0:y1, x0:x1]
    if frame.size == 0:
        raise ValueError("프레임 추출 결과가 비어 있습니다.")

    question_tag = f"question{question_idx:02d}"
    frame_name = f"frame_{question_tag}_{repeat_idx}_01.jpg"
    origin_name = f"frame_{question_tag}_{repeat_idx}_01_origin.jpg"
    frame_path = os.path.join(output_dir, frame_name)
    origin_path = os.path.join(output_dir, origin_name)
    cv2.imwrite(frame_path, frame)
    cv2.imwrite(origin_path, frame)
    print(f"1문자 프레임 저장됨: {frame_path}")

    debug = image.copy()
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
    debug_path = os.path.join(output_dir, "debug_one_character_frame.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"디버그 이미지 저장됨: {debug_path}")

    _create_template_output(frame, template_path, template_name)

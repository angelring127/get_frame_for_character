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

    debug_path = os.path.join(output_dir, "debug_four_character_checkboxes.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"체크박스 디버그 저장됨: {debug_path}")
    print(f"선택 결과: question{question_idx:02d}, repeat={repeat_idx}")
    return question_idx, repeat_idx


def _fallback_grid_boxes(image):
    h, w = image.shape[:2]
    x_margin = int(w * 0.045)
    y_top = int(h * 0.18)
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
        if y < int(h * 0.15) or y + bh > int(h * 0.92):
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
    top_row = sorted(selected[:2], key=lambda r: r[0])
    bottom_row = sorted(selected[2:], key=lambda r: r[0])
    return top_row + bottom_row


def four_character_detect_and_save_frames(image, output_dir, template_path=None, template_name=None):
    """
    Detect and save 4 large character boxes (2x2 layout) for four-character format.
    """
    if image is None or image.size == 0:
        raise ValueError("입력 이미지가 비어있거나 유효하지 않습니다.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    question_idx, repeat_idx = _detect_marked_question_repeat(image, output_dir)

    boxes = _detect_four_boxes(image)
    debug = image.copy()
    question_tag = f"question{question_idx:02d}"

    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        pad = max(4, int(min(bw, bh) * 0.01))
        x0, y0 = x + pad, y + pad
        x1, y1 = x + bw - pad, y + bh - pad
        frame = image[y0:y1, x0:x1]
        if frame.size == 0:
            continue

        suffix = f"{i:02d}"
        frame_name = f"frame_{question_tag}_{repeat_idx}_{suffix}.jpg"
        origin_name = f"frame_{question_tag}_{repeat_idx}_{suffix}_origin.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        cv2.imwrite(os.path.join(output_dir, origin_name), frame)

        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
        cv2.putText(debug, suffix, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    debug_path = os.path.join(output_dir, "debug_four_character_frames.jpg")
    cv2.imwrite(debug_path, debug)
    print(f"4문자 프레임 디버그 저장됨: {debug_path}")

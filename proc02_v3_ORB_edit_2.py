import cv2
import numpy as np

def distance_to_similarity(distance):
    if distance <= 10:  # 더 엄격한 거리 기준
        return 95 + (10 - distance) * 0.5  # 더 높은 기본 점수
    elif 10 < distance <= 20:
        return 95 - ((distance - 10) * (45 / 10))
    elif 20 < distance <= 30:
        return 50 - ((distance - 20) * (50 / 10))
    else:
        return 0

def preprocess_image(image, size=(200, 200)):
    # 이미지 크기 조정
    image = cv2.resize(image, size)
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 적응형 이진화
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # 노이즈 제거
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def extract_features(image):
    # ORB 특징점 추출 (SIFT보다 한자에 더 적합)
    orb = cv2.ORB_create(nfeatures=2000,
                         scaleFactor=1.2,
                         nlevels=8,
                         edgeThreshold=31,
                         firstLevel=0,
                         WTA_K=2,
                         patchSize=31)
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def match_features(des1, des2):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    
    # Brute Force 매처 사용
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # 거리 기반 필터링
    max_distance = 50
    good_matches = [m for m in matches if m.distance < max_distance]
    return sorted(good_matches, key=lambda x: x.distance)

def calculate_similarity(matches):
    if not matches:
        return 0
    
    # 상위 매칭 점수만 사용
    top_matches = matches[:min(30, len(matches))]
    distances = [m.distance for m in top_matches]
    avg_distance = np.mean(distances)
    
    # 시그모이드 함수로 유사도 변환
    similarity = 100 / (1 + np.exp(avg_distance/20 - 2))
    return similarity

def template_matching(img1, img2):
    # 이미지 크기 맞추기
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    
    # 여러 방법으로 템플릿 매칭 수행
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    scores = []
    
    for method in methods:
        res = cv2.matchTemplate(img1, img2, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        scores.append(max_val)
    
    # 시그모이드 함수로 점수 변환
    score = max(scores)
    normalized_score = 100 / (1 + np.exp(-12 * (score - 0.5)))
    return normalized_score

def extract_contours(image):
    # 윤곽선 추출
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_contour_similarity(contours1, contours2):
    if not contours1 or not contours2:
        return 0
    
    # 윤곽선 매칭
    total_similarity = 0
    matches = 0
    
    for cnt1 in contours1:
        best_match = 0
        for cnt2 in contours2:
            # 윤곽선 매칭 점수 계산
            match = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0.0)
            similarity = np.exp(-match * 20)  # 더 엄격한 변환 함수
            best_match = max(best_match, similarity)
        total_similarity += best_match
        matches += 1
    
    return (total_similarity / matches) * 100 if matches > 0 else 0

def calculate_hu_moments_similarity(image1, image2):
    # Hu 모멘트 계산
    moments1 = cv2.moments(image1)
    moments2 = cv2.moments(image2)
    
    hu1 = cv2.HuMoments(moments1)
    hu2 = cv2.HuMoments(moments2)
    
    # 로그 스케일 변환
    for i in range(7):
        if hu1[i] != 0:
            hu1[i] = -np.sign(hu1[i]) * np.log10(abs(hu1[i]))
        if hu2[i] != 0:
            hu2[i] = -np.sign(hu2[i]) * np.log10(abs(hu2[i]))
    
    # 유사도 계산 (더 엄격한 변환)
    similarity = 100 * np.exp(-np.sum(np.abs(hu1 - hu2)) / 3)
    return similarity

def calculate_stroke_similarity(img1, img2):
    # 세선화
    kernel = np.ones((3,3), np.uint8)
    img1_thin = cv2.erode(img1, kernel, iterations=1)
    img2_thin = cv2.erode(img2, kernel, iterations=1)
    
    # 획 특징 추출
    strokes1 = cv2.Canny(img1_thin, 50, 150)
    strokes2 = cv2.Canny(img2_thin, 50, 150)
    
    # 획수 비교
    stroke_count1 = np.sum(strokes1 > 0)
    stroke_count2 = np.sum(strokes2 > 0)
    
    count_ratio = min(stroke_count1, stroke_count2) / max(stroke_count1, stroke_count2)
    count_similarity = 100 * count_ratio
    
    # 획 위치 비교
    intersection = np.sum(np.logical_and(strokes1 > 0, strokes2 > 0))
    union = np.sum(np.logical_or(strokes1 > 0, strokes2 > 0))
    position_similarity = 100 * intersection / union if union > 0 else 0
    
    # 최종 획 유사도 (가중치 조정)
    return 0.2 * count_similarity + 0.8 * position_similarity

def calculate_zoning_similarity(img1, img2, zones=4):
    h, w = img1.shape
    h_step = h // zones
    w_step = w // zones
    
    similarities = []
    for i in range(zones):
        for j in range(zones):
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step
            
            zone1 = img1[y1:y2, x1:x2]
            zone2 = img2[y1:y2, x1:x2]
            
            # 구역별 픽셀 밀도 비교
            density1 = np.sum(zone1 > 0) / (h_step * w_step)
            density2 = np.sum(zone2 > 0) / (h_step * w_step)
            
            # 더 엄격한 유사도 계산
            similarity = 100 * np.exp(-abs(density1 - density2) * 8)
            similarities.append(similarity)
    
    return np.mean(similarities)

def calculate_kanji_similarity(img1, img2):
    # 전처리
    binary1 = preprocess_image(img1)
    binary2 = preprocess_image(img2)
    
    # 윤곽선 기반 유사도
    contours1 = extract_contours(binary1)
    contours2 = extract_contours(binary2)
    contour_score = calculate_contour_similarity(contours1, contours2)
    
    # Hu 모멘트 기반 유사도
    moment_score = calculate_hu_moments_similarity(binary1, binary2)
    
    # 획 기반 유사도
    stroke_score = calculate_stroke_similarity(binary1, binary2)
    
    # 구역 기반 유사도
    zoning_score = calculate_zoning_similarity(binary1, binary2)
    
    # 템플릿 매칭
    res = cv2.matchTemplate(binary1, binary2, cv2.TM_CCOEFF_NORMED)
    template_score = float(np.max(res)) * 100
    
    # 최종 점수 계산 (가중치 조정)
    weights = {
        'contour': 0.35,   # 윤곽선 유사도 가중치 증가
        'moment': 0.05,    # Hu 모멘트 가중치 감소
        'stroke': 0.45,    # 획 유사도 가중치 증가
        'zoning': 0.05,    # 구역 유사도 가중치 감소
        'template': 0.1    # 템플릿 매칭 가중치 유지
    }
    
    final_score = (
        contour_score * weights['contour'] +
        moment_score * weights['moment'] +
        stroke_score * weights['stroke'] +
        zoning_score * weights['zoning'] +
        template_score * weights['template']
    )
    
    # 결과 반환
    return {
        'final_score': final_score,
        'mse_score': contour_score,
        'stroke_score': stroke_score,
        'template_score': template_score,
        'structural_score': zoning_score
    }

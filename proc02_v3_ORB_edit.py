import cv2
import numpy as np

def distance_to_similarity(distance):
    if distance <= 20:
        return 80 + (20 - distance) * 1
    elif 20 < distance <= 40:
        return 80 - ((distance - 20) * (30 / 20))
    elif 40 < distance <= 50:
        return 50 - ((distance - 40) * (50 / 10))
    else:
        return 0

def preprocess_image(image):
    # 이미지 크기 정규화
    image = cv2.resize(image, (400, 400))
    
    # 이미지 개선
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 히스토그램 평활화
    
    # 노이즈 제거
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 적응형 이진화 사용
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def extract_features(image):
    # SIFT 사용 (더 강력한 특징점 검출)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    
    # 특징점이 너무 적으면 ORB도 시도
    if des is None or len(des) < 10:
        orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.1,
            nlevels=12,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        kp, des = orb.detectAndCompute(image, None)
    
    return kp, des

def match_features(des1, des2):
    if des1 is None or des2 is None:
        return []
    
    # SIFT의 경우 FLANN 매처 사용
    if des1.dtype.type is np.float32:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    else:
        # ORB의 경우 기존 브루트포스 매처 사용
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

def calculate_similarity(matches):
    if not matches:
        return 0
    
    # 매칭 점수 계산 방식 개선
    if len(matches) < 4:  # 매칭점이 너무 적으면 낮은 점수
        return len(matches) * 10
    
    # 상위 매칭점들의 거리를 기반으로 점수 계산
    top_matches = matches[:min(15, len(matches))]
    avg_distance = sum(match.distance for match in top_matches) / len(top_matches)
    
    # 거리를 0-100 스케일로 변환 (거리가 작을수록 유사도가 높음)
    similarity = max(0, 100 - (avg_distance * 0.8))
    return similarity

def calculate_overall_similarity(img1, img2):
    # 전처리
    proc1 = preprocess_image(img1)
    proc2 = preprocess_image(img2)
    
    # 특징점 매칭 기반 유사도
    kp1, des1 = extract_features(proc1)
    kp2, des2 = extract_features(proc2)
    matches = match_features(des1, des2)
    feature_similarity = calculate_similarity(matches)
    
    # 템플릿 매칭 기반 유사도
    template_similarity = template_matching(proc1, proc2)
    
    # 구조적 유사도 (SSIM) 추가
    ssim_score = cv2.matchTemplate(proc1, proc2, cv2.TM_CCOEFF_NORMED)[0][0] * 100
    
    # 가중치를 둔 최종 점수 계산
    final_score = (feature_similarity * 0.4 + 
                  template_similarity * 0.3 + 
                  ssim_score * 0.3)
    
    return final_score

def template_matching(img1, img2):
    # 템플릿 매칭 방식 개선
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    scores = []
    
    for method in methods:
        res = cv2.matchTemplate(img1, img2, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        scores.append(max_val * 100)
    
    return max(scores)  # 가장 높은 점수 반환

def preprocess_kanji(image):
    """한자 이미지 전처리"""
    # 크기 정규화
    image = cv2.resize(image, (64, 64))  # 더 작은 크기로 시작
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 테두리 추가 (패딩)
    binary = cv2.copyMakeBorder(binary, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    
    return binary

def get_contour_features(image):
    """윤곽선 특징 추출"""
    # 윤곽선 검출
    contours, _ = cv2.findContours(255 - image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 의미 있는 윤곽선만 필터링
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:  # 작은 노이즈 제거
            valid_contours.append(contour)
    
    return valid_contours

def calculate_kanji_similarity(img1, img2):
    """두 한자 이미지 비교"""
    # 전처리
    proc1 = preprocess_kanji(img1)
    proc2 = preprocess_kanji(img2)
    
    # 1. 픽셀 기반 유사도
    pixel_diff = np.mean(np.abs(proc1 - proc2))
    pixel_similarity = max(0, 100 * (1 - pixel_diff / 255))
    
    # 2. 윤곽선 특징 비교
    contours1 = get_contour_features(proc1)
    contours2 = get_contour_features(proc2)
    
    # 획수 유사도
    stroke_diff = abs(len(contours1) - len(contours2))
    if stroke_diff == 0:
        stroke_score = 100
    elif stroke_diff == 1:
        stroke_score = 50
    else:
        stroke_score = max(0, 100 - (stroke_diff * 20))
    
    # 3. 템플릿 매칭
    result = cv2.matchTemplate(proc1, proc2, cv2.TM_CCOEFF_NORMED)
    template_score = max(0, result[0][0] * 100)
    
    # 최종 점수 계산
    weights = {
        'pixel': 0.4,
        'stroke': 0.4,
        'template': 0.2
    }
    
    final_score = (
        pixel_similarity * weights['pixel'] +
        stroke_score * weights['stroke'] +
        template_score * weights['template']
    )
    
    # 임계값 기반 보정
    if stroke_score < 50:  # 획수가 많이 다르면
        final_score *= 0.5
    elif final_score > 90:  # 매우 유사하면
        final_score = 100
    
    return {
        'final_score': final_score,
        'pixel_similarity': pixel_similarity,
        'stroke_score': stroke_score,
        'template_score': template_score
    }

# 테스트 코드
if __name__ == "__main__":
    # 이미지 로드
    img1 = cv2.imread('question_02.png')
    img2 = cv2.imread('Sample03.jpg')
    
    if img1 is None or img2 is None:
        print("이미지를 불러올 수 없습니다.")
    else:
        results = calculate_kanji_similarity(img1, img2)
        
        print(f"최종 유사도 점수: {results['final_score']:.2f}%")
        print(f"픽셀 유사도: {results['pixel_similarity']:.2f}%")
        print(f"획수 유사도: {results['stroke_score']:.2f}%")
        print(f"템플릿 매칭: {results['template_score']:.2f}%")

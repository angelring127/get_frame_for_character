import cv2
import numpy as np

def resize_maintain_aspect(image, target_size=100):
    # 입력 이미지 검증
    if image is None or image.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 이미지의 높이와 너비 가져오기
    h, w = image.shape[:2]
    
    # 이미지가 너무 작은 경우 처리
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 더 긴 쪽을 기준으로 비율 계산
    ratio = target_size / float(max(h, w))
    
    # 새로운 크기 계산 (비율 유지)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    
    try:
        # 리사이즈
        resized = cv2.resize(image, new_size)
        
        # 패딩을 위한 새 이미지 생성
        square = np.zeros((target_size, target_size), dtype=np.uint8)
        if len(image.shape) == 3:
            square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 중앙에 리사이즈된 이미지 배치
        y_offset = (target_size - new_size[1]) // 2
        x_offset = (target_size - new_size[0]) // 2
        
        if len(image.shape) == 3:
            square[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0], :] = resized
        else:
            square[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
        
        return square
    except Exception as e:
        print(f"Resize error: {e}")
        return np.zeros((target_size, target_size), dtype=np.uint8)

def distance_to_similarity(distance):
    if distance <= 20:
        return 80 + (20 - distance) * 1
    elif 20 < distance <= 40:
        return 80 - ((distance - 20) * (30 / 20))
    elif 40 < distance <= 50:
        return 50 - ((distance - 40) * (50 / 10))
    else:
        return 0

def crop_character(image):
    # 입력 이미지 검증
    if image is None or image.size == 0:
        return image
    
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 이진화
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        # 가장 큰 윤곽선 찾기 (한자 영역으로 가정)
        main_contour = max(contours, key=cv2.contourArea)
        
        # 경계 상자 좌표 얻기
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # 너무 작은 영역은 원본 반환
        if w < 5 or h < 5:
            return image
        
        # 여백 추가 (10% 정도)
        padding = int(min(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # 한자 영역 추출
        cropped = image[y:y+h, x:x+w]
        
        # 추출된 영역이 너무 작으면 원본 반환
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            return image
            
        return cropped
    except Exception as e:
        print(f"Crop error: {e}")
        return image

def preprocess_image(image):
    # 한자 영역 추출
    cropped = crop_character(image)
    
    # 그레이스케일 변환
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped
    
    # 노이즈 제거를 위한 블러
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 적응형 이진화 적용
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def extract_features(image):
    # ORB 파라미터 조정
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        patchSize=31,
        fastThreshold=20
    )
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def match_features(des1, des2):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    except cv2.error:
        return []

def calculate_similarity(matches):
    if not matches:
        return 0
    
    # 상위 10개 매치 사용
    top_matches = matches[:10]
    avg_distance = sum(match.distance for match in top_matches) / len(top_matches)
    
    # 유사도 점수 계산 방식 개선
    similarity = max(0, 100 - (avg_distance * 2))
    return similarity

def template_matching(img1, img2):
    # 여러 크기로 템플릿 매칭 시도
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    max_score = -1
    
    for scale in scales:
        # 크기 조정
        width = int(img2.shape[1] * scale)
        height = int(img2.shape[0] * scale)
        resized = cv2.resize(img2, (width, height))
        
        try:
            result = cv2.matchTemplate(img1, resized, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            max_score = max(max_score, score)
        except:
            continue
    
    return max(0, max_score * 100)

def calculate_mse(img1, img2):
    # 비율 유지하며 이미지 크기 맞추기
    img1 = resize_maintain_aspect(img1)
    img2 = resize_maintain_aspect(img2)
    
    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    mse_score = max(0, 100 - (mse * 10))
    return mse_score

def calculate_stroke_similarity(img1, img2):
    # 획수 특징 추출 개선
    def extract_stroke_features(img):
        # 모멘트 계산
        moments = cv2.moments(img)
        
        # 획수 관련 특징들
        pixel_count = np.sum(img > 0)
        density = pixel_count / (img.shape[0] * img.shape[1])
        
        # 방향성 분포
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return [pixel_count, density, np.mean(gradient_magnitude)]
    
    # 특징 추출
    features1 = extract_stroke_features(img1)
    features2 = extract_stroke_features(img2)
    
    # 특징 간 유사도 계산
    differences = [abs(f1 - f2) / max(f1, f2) for f1, f2 in zip(features1, features2)]
    avg_diff = np.mean(differences)
    
    return max(0, 100 * (1 - avg_diff))

def calculate_structural_similarity(img1, img2):
    # 구조적 특징 추출 개선
    def extract_structural_features(img):
        # 윤곽선 검출
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # 주요 윤곽선
        main_contour = max(contours, key=cv2.contourArea)
        
        # 특징 계산
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w)/h if h != 0 else 0
        extent = float(area)/(w*h) if w*h != 0 else 0
        
        return [area, perimeter, aspect_ratio, extent]
    
    # 특징 추출
    features1 = extract_structural_features(img1)
    features2 = extract_structural_features(img2)
    
    if features1 is None or features2 is None:
        return 0
    
    # 특징 간 유사도 계산
    differences = [abs(f1 - f2) / max(abs(f1), abs(f2), 1) for f1, f2 in zip(features1, features2)]
    avg_diff = np.mean(differences)
    
    return max(0, 100 * (1 - avg_diff))

def calculate_kanji_similarity(img1, img2):
    # 전처리
    binary1 = preprocess_image(img1)
    binary2 = preprocess_image(img2)
    
    # 특징량 추출 및 매칭
    kp1, des1 = extract_features(binary1)
    kp2, des2 = extract_features(binary2)
    matches = match_features(des1, des2)
    
    # 각각의 유사도 점수 계산
    orb_score = calculate_similarity(matches)
    template_score = template_matching(binary1, binary2)
    mse_score = calculate_mse(binary1, binary2)
    stroke_score = calculate_stroke_similarity(binary1, binary2)
    structural_score = calculate_structural_similarity(binary1, binary2)
    
    # 가중치 조정
    final_score = (
        orb_score * 0.25 +
        template_score * 0.35 +
        mse_score * 0.15 +
        stroke_score * 0.15 +
        structural_score * 0.1
    )
    
    return {
        'final_score': final_score,
        'mse_score': mse_score,
        'stroke_score': stroke_score,
        'template_score': template_score,
        'structural_score': structural_score
    }

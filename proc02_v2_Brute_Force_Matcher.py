import cv2
import numpy as np
import urllib.request
import sys

def url_to_image(url):
    """URL에서 이미지를 다운로드하여 OpenCV 형식으로 변환"""
    try:
        # URL에서 이미지 다운로드
        resp = urllib.request.urlopen(url)
        # 바이트 배열로 변환
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # OpenCV 이미지로 디코딩
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"URL에서 이미지를 불러오는데 실패했습니다: {e}")
        return None

def calculate_structural_similarity(img1, img2):
    """구조적 유사도를 계산하는 함수"""
    # 이미지 크기가 같아야 함
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 1. 전체 픽셀 기반 유사도 (20%)
    pixel_similarity = 1 - np.sum(cv2.absdiff(img1, img2)) / (img1.shape[0] * img1.shape[1] * 255)
    
    # 2. 지역적 특징 비교 (30%)
    def compare_regions(img1, img2):
        h, w = img1.shape
        regions = []
        # 이미지를 9개 영역으로 나누어 비교
        for i in range(3):
            for j in range(3):
                r1 = img1[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                r2 = img2[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                diff = 1 - np.sum(cv2.absdiff(r1, r2)) / (r1.shape[0] * r1.shape[1] * 255)
                regions.append(diff)
        return np.mean(regions)
    
    region_similarity = compare_regions(img1, img2)
    
    # 3. 윤곽선 특징 비교 (30%)
    edges1 = cv2.Canny(img1, 50, 150)
    edges2 = cv2.Canny(img2, 50, 150)
    
    # 윤곽선 매칭
    contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 윤곽선 개수 비교
    contour_count_similarity = min(len(contours1), len(contours2)) / max(len(contours1), len(contours2))
    
    # 윤곽선 모양 비교
    edge_similarity = 1 - np.sum(cv2.absdiff(edges1, edges2)) / (edges1.shape[0] * edges1.shape[1] * 255)
    contour_similarity = (contour_count_similarity + edge_similarity) / 2
    
    # 4. 획의 방향성 분포 (20%)
    sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
    
    # 방향성 히스토그램 계산
    magnitude1 = np.sqrt(sobelx1**2 + sobely1**2)
    magnitude2 = np.sqrt(sobelx2**2 + sobely2**2)
    direction1 = np.arctan2(sobely1, sobelx1)
    direction2 = np.arctan2(sobely2, sobelx2)
    
    # 방향성 히스토그램 비교 (8방향)
    hist1 = np.zeros(8)
    hist2 = np.zeros(8)
    for i in range(8):
        mask1 = (direction1 >= i*np.pi/4) & (direction1 < (i+1)*np.pi/4) & (magnitude1 > 30)
        mask2 = (direction2 >= i*np.pi/4) & (direction2 < (i+1)*np.pi/4) & (magnitude2 > 30)
        hist1[i] = np.sum(mask1)
        hist2[i] = np.sum(mask2)
    
    # 정규화
    hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1
    hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2
    direction_similarity = 1 - np.sum(np.abs(hist1 - hist2)) / 2
    
    # 최종 유사도 계산 (가중치 적용)
    final_similarity = (
        0.2 * pixel_similarity +
        0.3 * region_similarity +
        0.3 * contour_similarity +
        0.2 * direction_similarity
    ) * 100
    
    # 엄격한 임계값 적용
    if final_similarity > 80:
        # 매우 높은 유사도의 경우 더 엄격한 검증
        if (pixel_similarity < 0.7 or 
            region_similarity < 0.7 or 
            contour_similarity < 0.7 or 
            direction_similarity < 0.7):
            final_similarity *= 0.7
    
    return final_similarity

def find_character_box(image):
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 가장 큰 윤곽선 찾기
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 여백 추가 (15%)
    padding = int(max(w, h) * 0.15)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - x, w + 2 * padding)
    h = min(gray.shape[0] - y, h + 2 * padding)
    
    return (x, y, w, h)

def preprocess_image(image):
    # 그레이스케일로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 이미지 크기 정규화 (큰 크기로)
    target_size = 800
    aspect_ratio = gray.shape[1] / gray.shape[0]
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    gray = cv2.resize(gray, (new_width, new_height))

    # 패딩 추가
    top = bottom = int(new_height * 0.1)
    left = right = int(new_width * 0.1)
    gray = cv2.copyMakeBorder(gray, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=255)

    # 대비 향상 (더 강화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 노이즈 제거를 위한 bilateral 필터 적용
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 추가 대비 향상
    enhanced = cv2.equalizeHist(denoised)

    # 적응형 이진화 적용
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5
    )

    # 모폴로지 연산으로 노이즈 제거 및 글자 보완
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("윤곽선을 찾을 수 없습니다.")
        return None

    # 면적이 특정 임계값 이상인 윤곽선만 선택
    min_area = 100  # 최소 면적 임계값
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not valid_contours:
        print("유효한 윤곽선을 찾을 수 없습니다.")
        return None

    # 모든 유효한 윤곽선을 포함하는 경계 상자 찾기
    x_min = min([cv2.boundingRect(cnt)[0] for cnt in valid_contours])
    y_min = min([cv2.boundingRect(cnt)[1] for cnt in valid_contours])
    x_max = max([cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in valid_contours])
    y_max = max([cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in valid_contours])

    # 패딩 추가 (25%)
    padding_x = int((x_max - x_min) * 0.25)
    padding_y = int((y_max - y_min) * 0.25)
    
    # 패딩을 포함한 영역이 이미지 경계를 벗어나지 않도록 조정
    x1 = max(0, x_min - padding_x)
    y1 = max(0, y_min - padding_y)
    x2 = min(gray.shape[1], x_max + padding_x)
    y2 = min(gray.shape[0], y_max + padding_y)

    # 글자 영역 추출
    char_region = gray[y1:y2, x1:x2]

    # 최종 크기 조정
    final_size = 400
    char_region = cv2.resize(char_region, (final_size, final_size))

    # 최종 이진화
    _, final_binary = cv2.threshold(char_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 디버그용 이미지 표시
    # cv2.imshow('Original Gray', gray)
    # cv2.imshow('Enhanced', enhanced)
    # cv2.imshow('Binary', binary)
    # cv2.imshow('Char Region', char_region)
    # cv2.imshow('Final Binary', final_binary)
    # cv2.waitKey(1)

    return final_binary

def main(path1, path2):
    """
    한자 이미지 유사도 비교 메인 함수
    Args:
        path1: 첫 번째 이미지 경로 (URL 또는 로컬 파일 경로)
        path2: 두 번째 이미지 경로 (로컬 파일 경로)
    """
    # 이미지 로드
    if path1.startswith('http'):
        img1 = url_to_image(path1)
    else:
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 이미지 전처리
    processed1 = preprocess_image(img1)
    processed2 = preprocess_image(img2)

    if processed1 is None or processed2 is None:
        print("한자 영역을 찾을 수 없습니다.")
        return

    # 구조적 유사도 계산
    similarity = calculate_structural_similarity(processed1, processed2)

    print(f'{similarity:.2f}')

    # # 결과 시각화
    # cv2.imshow('Processed Image 1', processed1)
    # cv2.imshow('Processed Image 2', processed2)

    # # 두 이미지를 나란히 표시
    # combined = np.hstack((processed1, processed2))
    # cv2.imshow('Comparison', combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return similarity

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python proc02_v2_Brute_Force_Matcher.py <path1/url1> <path2>")
        sys.exit(1)
    
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    
    main(path1, path2)

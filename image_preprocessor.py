import cv2
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self):
        self.max_dimension = 1500
        self.A4_RATIO = 1.4142  # A4 용지 비율 (1:√2)

    def remove_shadows(self, image):
        """
        이미지에서 그림자를 제거합니다.
        """
        # RGB를 LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # L 채널에 대해 CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # 밝기 채널 조정
        cl = cv2.normalize(cl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # 채널 병합 및 BGR로 변환
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 추가 그림자 제거를 위한 블렌딩
        result = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        
        return result

    def resize_image(self, image):
        """
        이미지 크기를 조정합니다.
        """
        scale = 1.0
        if max(image.shape[0], image.shape[1]) > self.max_dimension:
            scale = self.max_dimension / max(image.shape[0], image.shape[1])
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            image = cv2.resize(image, (width, height))
        return image, scale

    def enhance_image(self, image):
        """
        이미지의 품질을 개선합니다.
        특히 격자 구조가 잘 보이도록 대비를 강화하고 노이즈를 제거합니다.
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거 (약하게 조정)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 적응형 이진화로 대비 강화 (파라미터 미세 조정)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 13, 4  # 블록 크기와 C값 미세 조정
        )
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((2,2), np.uint8)  # 커널 크기 감소
        
        # 열림 연산으로 작은 점 제거 (반복 횟수 감소)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 닫힘 연산으로 글자 영역 보존
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 중간값 필터로 추가 노이즈 제거 (커널 크기 감소)
        binary = cv2.medianBlur(binary, 3)
        
        # 작은 컴포넌트 제거
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # 작은 컴포넌트 제거 (면적 기준 감소)
        min_size = 30  # 최소 면적 기준 감소
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                binary[labels == i] = 255
        
        # 결과를 3채널로 변환
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return enhanced

    def preprocess(self, image_path, remove_shadow=True):
        """
        이미지에 대한 모든 전처리를 수행합니다.
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
            
        # 디버깅을 위한 디렉토리 생성
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join("extracted", base_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"디버그 이미지 저장 디렉토리 생성: {output_dir}")
        
        # 원본 이미지 저장
        cv2.imwrite(os.path.join(output_dir, 'debug_original.jpg'), image)
        print(f"원본 이미지 저장됨: {output_dir}/debug_original.jpg")
            
        # 크기 조정 (그림자 제거 전에 수행)
        image, scale = self.resize_image(image)
            
        # 그림자 제거
        if remove_shadow:
            image = self.remove_shadows(image)
            
        # 이미지 품질 개선
        enhanced = self.enhance_image(image)
        
        # 중간 과정 이미지 저장
        cv2.imwrite(os.path.join(output_dir, 'debug_resized.jpg'), image)
        cv2.imwrite(os.path.join(output_dir, 'debug_enhanced.jpg'), enhanced)
        print(f"크기 조정된 이미지 저장됨: {output_dir}/debug_resized.jpg")
        print(f"전처리된 이미지 저장됨: {output_dir}/debug_enhanced.jpg")
        
        return enhanced, scale

    def detect_paper(self, image, image_path):
        """
        이미지에서 A4 용지 영역을 감지합니다.
        A4 용지의 특성(직사각형, 1:√2 비율)을 활용합니다.
        """
        # 디버그 이미지를 위한 복사본 생성
        debug_image = image.copy()
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러로 노이즈 제거 (커널 크기 감소)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny 엣지 검출 추가
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 모폴로지 연산으로 엣지 강화
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 면적 기준으로 필터링된 윤곽선 찾기
        valid_contours = []
        img_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > img_area * 0.3:  # 이미지 면적의 30% 이상인 윤곽선만 선택
                valid_contours.append(contour)
        
        if valid_contours:
            # 면적이 가장 큰 윤곽선 선택
            page_contour = max(valid_contours, key=cv2.contourArea)
            
            # 윤곽선 근사화 (epsilon 값 증가)
            epsilon = 0.02 * cv2.arcLength(page_contour, True)  # 더 정밀하게
            approx = cv2.approxPolyDP(page_contour, epsilon, True)
            
            # 디버그용 윤곽선 그리기
            cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 2)
            
            # 근사화된 윤곽선이 4개의 꼭지점을 가지면 종이로 간주
            if len(approx) == 4:
                # 꼭지점 순서 정렬 (좌상단부터 시계방향)
                points = approx.reshape(4, 2)
                
                # x+y 값이 가장 작은 점이 좌상단
                top_left = points[np.argmin(np.sum(points, axis=1))]
                # x+y 값이 가장 큰 점이 우하단
                bottom_right = points[np.argmax(np.sum(points, axis=1))]
                
                # x-y 값이 가장 작은 점이 좌하단
                bottom_left = points[np.argmin(points[:,0] - points[:,1])]
                # x-y 값이 가장 큰 점이 우상단
                top_right = points[np.argmax(points[:,0] - points[:,1])]
                
                # 정렬된 점들
                points = np.array([top_left, top_right, bottom_right, bottom_left])
                
                # A4 비율 검증
                width = np.linalg.norm(points[0] - points[1])
                height = np.linalg.norm(points[1] - points[2])
                ratio = max(width, height) / min(width, height)
                
                # A4 비율(1.4142)과의 오차가 20% 이내인 경우에만 사용
                if 1.1 < ratio < 1.7:
                    # 디버그용 꼭지점 그리기
                    for point in points:
                        cv2.circle(debug_image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
                    
                    # 디버그 이미지 저장
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_dir = os.path.join("extracted", base_name)
                    cv2.imwrite(os.path.join(output_dir, 'debug_paper_detection.jpg'), debug_image)
                    print(f"종이 감지 디버그 이미지 저장됨: {output_dir}/debug_paper_detection.jpg")
                    
                    return points.astype(np.int32)
        
        print("A4 용지를 찾지 못했습니다. 문서 크롭을 건너뜁니다.")
        return None

    def crop_and_transform(self, image, box, scale, image_path):
        """
        감지된 문서 영역을 크롭하고 원근 변환을 적용합니다.
        
        Args:
            image: 원본 이미지
            box: 감지된 문서의 네 꼭지점 좌표
            scale: 이미지 크기 조정 비율
            image_path: 원본 이미지 경로 (디버그 이미지 저장용)
            
        Returns:
            변환된 이미지
        """
        # 원본 이미지 복사본 생성
        debug_image = image.copy()
        
        # 꼭지점 좌표를 원본 이미지 크기에 맞게 조정
        box = box / scale
        
        # 문서의 너비와 높이 계산
        width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))
        
        # 목적지 좌표 설정 (A4 비율 유지)
        dst_points = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)
        
        # 원근 변환 행렬 계산
        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), dst_points)
        
        # 원근 변환 적용
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        # 디버그 이미지에 변환된 영역 표시
        cv2.polylines(debug_image, [box.astype(np.int32)], True, (0, 255, 0), 2)
        
        # 디버그 이미지 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join("extracted", base_name)
        cv2.imwrite(os.path.join(output_dir, 'debug_crop_transform.jpg'), debug_image)
        print(f"문서 크롭 및 변환 디버그 이미지 저장됨: {output_dir}/debug_crop_transform.jpg")
        
        return warped

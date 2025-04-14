from PIL import Image
import os
import numpy as np
import json
import requests
import uuid
import time
import argparse

def clean_image(img):
    """
    이미지를 깔끔하게 만듭니다.
    - 배경을 완전한 흰색으로 만듭니다
    - 테두리의 잔여물을 제거합니다
    """
    # 이미지를 numpy 배열로 변환
    img_array = np.array(img)
    
    # RGB 값이 모두 210 이상인 픽셀을 완전한 흰색(255, 255, 255)으로 변환
    white_mask = np.all(img_array > 210, axis=2)
    img_array[white_mask] = [255, 255, 255]
    
    # 테두리 부분을 흰색으로 설정 (더 넓은 범위)
    img_array[0:4, :] = [255, 255, 255]  # 위쪽 테두리
    img_array[-4:, :] = [255, 255, 255]  # 아래쪽 테두리
    img_array[:, 0:4] = [255, 255, 255]  # 왼쪽 테두리
    img_array[:, -4:] = [255, 255, 255]  # 오른쪽 테두리
    
    return Image.fromarray(img_array)

def get_content_size(img_array):
    """
    이미지에서 실제 콘텐츠(글자)가 차지하는 크기를 반환합니다.
    """
    non_white = np.where(np.any(img_array != 255, axis=2))
    if len(non_white[0]) > 0:
        height = non_white[0].max() - non_white[0].min()
        width = non_white[1].max() - non_white[1].min()
        return height, width
    return 0, 0

def trim_image(img, target_padding=None):
    """
    이미지의 흰색 여백을 제거하고 모든 이미지가 동일한 여백을 가지도록 조정합니다.
    
    Args:
        img: PIL Image 객체
        target_padding: 목표로 하는 여백 크기 (None인 경우 자동 계산)
    """
    img_array = np.array(img)
    
    # 흰색이 아닌 픽셀의 위치 찾기
    non_white = np.where(np.any(img_array != 255, axis=2))
    
    if len(non_white[0]) > 0:
        # 글자 영역 찾기
        top = non_white[0].min()
        bottom = non_white[0].max() + 1
        left = non_white[1].min()
        right = non_white[1].max() + 1
        
        # 여백 크기 계산
        if target_padding is None:
            content_height = bottom - top
            content_width = right - left
            vertical_padding = (img_array.shape[0] - content_height) // 2
            horizontal_padding = (img_array.shape[1] - content_width) // 2
            # 여백 크기를 원본의 1/4로 줄임
            target_padding = max(vertical_padding, horizontal_padding) // 4
        
        # 새로운 크기 계산
        new_height = (bottom - top) + 2 * target_padding
        new_width = (right - left) + 2 * target_padding
        
        # 새로운 이미지 생성
        new_img = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))
        # 원본 글자 영역 복사
        content = Image.fromarray(img_array[top:bottom, left:right])
        # 중앙에 붙이기
        new_img.paste(content, (target_padding, target_padding))
        return new_img
    return img

def concatenate_images_horizontally(image_paths, output_path, spacing=-2):  # spacing을 음수로 설정하여 글자들을 더 가깝게
    """
    주어진 이미지들을 가로로 이어붙여 하나의 이미지로 만듭니다.
    """
    # 이미지들을 열어서 전처리
    images = []
    max_padding = 0
    
    # 첫 번째 패스: 모든 이미지를 처리하고 최대 여백 찾기
    for path in image_paths:
        img = Image.open(path)
        img = clean_image(img)
        img_array = np.array(img)
        height, width = get_content_size(img_array)
        vertical_padding = (img_array.shape[0] - height) // 2
        horizontal_padding = (img_array.shape[1] - width) // 2
        max_padding = max(max_padding, vertical_padding, horizontal_padding)
        images.append(img)
    
    # 최대 여백을 1/4로 줄임
    max_padding = max_padding // 4
    
    # 두 번째 패스: 동일한 여백으로 이미지 처리
    processed_images = [trim_image(img, max_padding) for img in images]
    
    # 전체 너비 계산
    total_width = sum(img.width for img in processed_images) + spacing * (len(processed_images) - 1)
    max_height = max(img.height for img in processed_images)
    
    # 결과 이미지 생성
    result = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    
    # 이미지들을 가로로 이어붙이기
    current_width = 0
    for i, img in enumerate(processed_images):
        # 이미지의 세로 중앙 정렬을 위한 y 좌표 계산
        y = (max_height - img.height) // 2
        # 이미지 붙이기
        result.paste(img, (current_width, y))
        # 다음 이미지의 시작 위치 업데이트
        current_width += img.width + (spacing if i < len(processed_images) - 1 else 0)
    
    # 결과 이미지 저장
    result.save(output_path, quality=100)
    print(f"이미지가 성공적으로 저장되었습니다: {output_path}")

def perform_ocr(image_path):
    """
    이미지 파일을 OCR API로 전송하여 텍스트를 추출합니다.
    
    Args:
        image_path (str): OCR을 수행할 이미지 파일의 경로
        
    Returns:
        str: 추출된 텍스트
    """
    url = "https://ocr-api.userlocal.jp/recognition/cropped"
    
    try:
        with open(image_path, "rb") as img:
            data = {"imgData": img}
            response = requests.post(url, files=data)
            
            if response.status_code == 200:
                result = json.loads(response.content)
                return result.get('text', '')
            else:
                print(f"API 요청 실패: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

def process_images(image_paths):
    """
    주어진 이미지 파일들을 처리하여 OCR 결과를 반환합니다.
    
    Args:
        image_paths (list): 처리할 이미지 파일 경로 리스트
        
    Returns:
        str: OCR로 추출된 텍스트
    """
    # 랜덤 파일명 생성
    random_filename = f"temp_{uuid.uuid4().hex[:8]}_{int(time.time())}.jpg"
    output_path = random_filename
    
    try:
        # 이미지 이어붙이기 실행
        concatenate_images_horizontally(image_paths, output_path)
        
        # OCR 수행
        print("\nOCR 결과:")
        result = perform_ocr(output_path)
        if result:
            print(f"추출된 텍스트: {result}")
            return result
        else:
            print("텍스트 추출 실패")
            return None
    finally:
        # 임시 파일 삭제
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"\n임시 파일이 삭제되었습니다: {output_path}")

if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='이미지들을 이어붙이고 OCR을 수행합니다.')
    parser.add_argument('image_paths', nargs='+', help='처리할 이미지 파일들의 경로')
    args = parser.parse_args()
    
    # 이미지 파일 경로 검증
    for path in args.image_paths:
        if not os.path.exists(path):
            print(f"오류: 파일을 찾을 수 없습니다: {path}")
            exit(1)
    
    # 이미지 처리 실행
    result = process_images(args.image_paths)
    if result:
        print(f"\nfinal_ocr_result: {result}")
    else:
        print("\nOCR 처리 실패") 
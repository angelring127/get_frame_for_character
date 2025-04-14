import json
import requests

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

if __name__ == "__main__":
    # 테스트용 이미지 경로
    test_image = "concatenated_image.jpg"
    result = perform_ocr(test_image)
    
    if result:
        print(f"추출된 텍스트: {result}")
    else:
        print("텍스트 추출 실패") 
from writing_detector import writing_detect_and_save_frames
from reading_detector import reading_detect_and_save_frames

def main():
    # 이미지 파일 경로와 출력 디렉토리 설정
    image_path = "reading_4_characters_03.jpg"
    output_dir = "extracted_frames"
    
    # 이미지 파일명에 따라 적절한 함수 호출
    if "reading" in image_path:
        reading_detect_and_save_frames(image_path, output_dir)
    elif "writing" in image_path:
        writing_detect_and_save_frames(image_path, output_dir)
    else:
        print("이미지 파일명에 'reading' 또는 'writing'이 포함되어 있지 않습니다.")

if __name__ == "__main__":
    main()
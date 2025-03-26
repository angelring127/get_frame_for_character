from writing_detector import writing_detect_and_save_frames
from reading_detector import reading_detect_and_save_frames
import os

def main(image_path: str, flag: int):
    # 이미지 파일명에서 확장자를 제외한 이름을 추출하여 출력 디렉토리 설정
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"{base_name}_extracted"
    
    # 플래그 값에 따라 적절한 함수 호출
    if flag == 1:
        writing_detect_and_save_frames(image_path, output_dir)
    elif flag == 2:
        reading_detect_and_save_frames(image_path, output_dir)
    else:
        print("잘못된 플래그 값입니다. 1(writing) 또는 2(reading)를 사용하세요.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("사용법: python main.py <이미지_경로> <플래그>")
        print("플래그: 1=writing, 2=reading")
        sys.exit(1)
        
    image_path = sys.argv[1]
    flag = int(sys.argv[2])
    main(image_path, flag)
import os
import cv2
from reading_detector import reading_detect_and_save_frames
from writing_detector import writing_detect_and_save_frames
from image_preprocessor import ImagePreprocessor

def main(image_path, flag, template_name=None):
    # 이미지 파일명에서 확장자를 제외한 이름을 추출하여 출력 디렉토리 설정
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # extracted 디렉토리 생성
    if not os.path.exists("extracted"):
        os.makedirs("extracted")
    
    # 출력 디렉토리 설정 및 생성
    output_dir = os.path.join("extracted", base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # output 디렉토리 생성 (템플릿 결과 저장용)
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 템플릿 파일 경로 설정 (.png와 .jpg 확장자 모두 확인)
    template_png = f"{template_name}.png" if template_name else None
    template_jpg = f"{template_name}.jpg" if template_name else None
    template_path = None
    
    if template_name:
        # 먼저 output 디렉토리에서 템플릿 파일 확인
        output_template_png = os.path.join("output", template_png)
        output_template_jpg = os.path.join("output", template_jpg)
        
        if os.path.exists(output_template_png):
            template_path = output_template_png
            print(f"템플릿 파일을 output 디렉토리에서 찾았습니다: {output_template_png}")
        elif os.path.exists(output_template_jpg):
            template_path = output_template_jpg
            print(f"템플릿 파일을 output 디렉토리에서 찾았습니다: {output_template_jpg}")
        # 루트 디렉토리에서 템플릿 파일 확인
        elif os.path.exists(template_png):
            template_path = template_png
        elif os.path.exists(template_jpg):
            template_path = template_jpg
        else:
            # 기본 템플릿 파일 사용
            if os.path.exists("template.png"):
                print(f"지정한 템플릿 파일을 찾을 수 없습니다. 기본 템플릿 파일을 사용하되 출력은 '{template_name}'으로 저장합니다.")
                template_path = "template.png"
            else:
                print("템플릿 파일을 찾을 수 없습니다.")
    
    # 이미지 전처리기 초기화
    preprocessor = ImagePreprocessor()
    
    try:
        # 이미지 전처리 수행
        processed_image, scale = preprocessor.preprocess(image_path)
        
        # 종이 영역 감지
        # box = preprocessor.detect_paper(processed_image, image_path)
        # if box is not None:
        #     # 종이 영역 자르기 및 변환
        #     processed_image = preprocessor.crop_and_transform(processed_image, box, scale, image_path)
        #     # 디버깅을 위해 크롭된 이미지 저장
        #     debug_cropped_path = os.path.join(output_dir, 'debug_cropped.jpg')
        #     cv2.imwrite(debug_cropped_path, processed_image)
        #     print(f"크롭된 이미지 저장됨: {debug_cropped_path}")
        # else:
        #     print("문서 감지에 실패하여 크롭을 건너뜁니다.")
        
        # 플래그 값에 따라 적절한 함수 호출
        if flag == 1:
            writing_detect_and_save_frames(processed_image, output_dir, template_path, template_name)
        elif flag == 2:
            reading_detect_and_save_frames(processed_image, output_dir, template_path, template_name)
        else:
            print("잘못된 플래그 값입니다. 1(writing) 또는 2(reading)를 사용하세요.")
            return
            
        print("처리가 완료되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python main.py <이미지_경로> <플래그> [템플릿_이름]")
        print("플래그: 1=writing, 2=reading")
        sys.exit(1)
    
    image_path = sys.argv[1]
    flag = int(sys.argv[2])  # 문자열을 정수로 변환
    template_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    main(image_path, flag, template_name)
from writing_detector import writing_detect_and_save_frames
from reading_detector import reading_detect_and_save_frames
import os

def main(image_path: str, flag: int, template_name: str):
    # 이미지 파일명에서 확장자를 제외한 이름을 추출하여 출력 디렉토리 설정
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = f"{base_name}_extracted"
    
    # output 디렉토리 생성
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 템플릿 파일 경로 설정 (.png와 .jpg 확장자 모두 확인)
    template_png = f"{template_name}.png"
    template_jpg = f"{template_name}.jpg"
    
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
            print(f"지정한 템플릿 파일 '{template_png}' 또는 '{template_jpg}'이 없습니다. 기본 템플릿 파일을 사용하되 출력은 '{template_name}'으로 저장합니다.")
            template_path = "template.png"
            # template_name은 변경하지 않음
        else:
            print("템플릿 파일을 찾을 수 없습니다.")
            template_path = None
            template_name = None
    
    # 플래그 값에 따라 적절한 함수 호출
    if flag == 1:
        writing_detect_and_save_frames(image_path, output_dir, template_path, template_name)
    elif flag == 2:
        reading_detect_and_save_frames(image_path, output_dir, template_path, template_name)
    else:
        print("잘못된 플래그 값입니다. 1(writing) 또는 2(reading)를 사용하세요.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("사용법: python main.py <이미지_경로> <플래그> <템플릿_이름>")
        print("플래그: 1=writing, 2=reading")
        print("템플릿_이름: 확장자(.png)를 제외한 템플릿 파일 이름")
        sys.exit(1)
        
    image_path = sys.argv[1]
    flag = int(sys.argv[2])
    template_name = sys.argv[3]
    
    main(image_path, flag, template_name)
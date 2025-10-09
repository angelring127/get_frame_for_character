from PIL import Image, ImageDraw, ImageFont

def create_template_with_frames():
    # 5000x5000 크기의 흰색 이미지 생성
    width = 5000
    height = 5000
    white_image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(white_image)
    
    # 프레임 크기와 간격 설정
    frame_size = 130
    
    # 23행으로 설정
    rows = 33
    cols = 34  # 첫 줄에 1부터 22까지의 숫자를 넣기 위해 22열로 설정
    
    # 실제 사용할 프레임 개수
    total_frames = rows * cols
    
    # 시작 위치 계산 (중앙 정렬)
    start_x = (width - (cols * (frame_size))) // 2
    start_y = (height - (rows * (frame_size))) // 2
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("Arial", 40)
    except:
        font = ImageFont.load_default()
    
    # 프레임 그리기
    frame_count = 0
    for row in range(rows):
        for col in range(cols):
            # 7번째 줄부터는 4칸 합치고 1칸 빈칸 패턴으로 처리
            if row >= 6:  # 7번째 줄 (인덱스 6)
                # 5칸을 한 그룹으로 봄 (4칸 합친 프레임 + 1칸 빈칸)
                position_in_group = col % 5
                
                # 그룹의 첫 4칸은 합쳐진 프레임, 5번째 칸은 빈칸
                if position_in_group == 0:  # 각 그룹의 시작 위치에만 합쳐진 프레임 그리기
                    # 4칸을 합친 큰 프레임의 좌상단 좌표
                    x1 = start_x + col * (frame_size)
                    y1 = start_y + row * (frame_size)
                    # 4칸을 합친 큰 프레임의 우하단 좌표
                    x2 = x1 + (4 * frame_size)
                    y2 = y1 + frame_size
                    
                    # 큰 프레임의 외곽 테두리만 그리기 (두께 2픽셀)
                    draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
                    frame_count += 1
                elif position_in_group == 4:  # 빈칸은 개별 프레임으로 표시
                    # 빈칸의 좌상단 좌표
                    x1 = start_x + col * (frame_size)
                    y1 = start_y + row * (frame_size)
                    # 빈칸의 우하단 좌표
                    x2 = x1 + frame_size
                    y2 = y1 + frame_size
                    
                    # 빈칸도 테두리 그리기 (두께 2픽셀)
                    draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
                    frame_count += 1
            else:
                # 1-6번째 줄은 기존 방식으로 개별 프레임 그리기
                # 프레임의 좌상단 좌표
                x1 = start_x + col * (frame_size)
                y1 = start_y + row * (frame_size)
                # 프레임의 우하단 좌표
                x2 = x1 + frame_size
                y2 = y1 + frame_size
                
                # 검은색 테두리 그리기 (두께 2픽셀)
                draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
                
                # 첫 줄에 1부터 34까지의 숫자 추가
                if row == 0:
                    number = col + 1
                    # 숫자를 프레임 중앙에 배치
                    text = str(number)
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = x1 + (frame_size - text_width) // 2
                    text_y = y1 + (frame_size - text_height) // 2
                    draw.text((text_x, text_y), text, fill='black', font=font)
                
                frame_count += 1
    
    # 이미지 저장
    white_image.save('template_with_frames.png')
    print(f"{width}x{height} 크기의 이미지에 {rows}x{cols}개의 프레임이 추가되었습니다.")
    print(f"총 {frame_count}개의 프레임이 그려졌습니다.")

if __name__ == "__main__":
    create_template_with_frames()

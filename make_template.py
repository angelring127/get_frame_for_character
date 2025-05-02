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
            if frame_count >= total_frames:
                break
                
            # 프레임의 좌상단 좌표
            x1 = start_x + col * (frame_size)
            y1 = start_y + row * (frame_size)
            # 프레임의 우하단 좌표
            x2 = x1 + frame_size
            y2 = y1 + frame_size
            
            # 검은색 테두리 그리기 (두께 2픽셀)
            draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
            
            # 첫 줄에 1부터 22까지의 숫자 추가
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
